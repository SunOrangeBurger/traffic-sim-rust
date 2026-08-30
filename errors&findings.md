# Errors & Findings Log

Running log of what was investigated, what broke, and what was fixed across
the gridlock-filter and training-diagnostics work on `traffic-sim-rust`.
Written for handoff — each entry states what was found, how it was verified,
and what changed as a result. Companion to `progress_so_far.md`, which has
the higher-level project narrative; this doc is the detailed error trail.

---

## 1. City generation & simulation review (initial pass)

**No errors found.** Reviewed `core/src/city.rs` and `core/src/sim.rs`
against their own test suite (`same_seed_same_city`,
`different_seed_likely_different_city`, `zone_weights_are_bounded_and_peak_near_hubs`,
`arterial_roads_cluster_near_hubs`) and confirmed the procedural generator is
deterministic, varied, and structured (arterial roads correlate with hub
proximity, not spatial noise). Cross-machine byte-identical sweep output
(noted in `progress_so_far.md`) independently confirms determinism.

**Finding, not an error:** city generation being genuinely varied is a
double-edged property. The same mechanism that makes cities feel realistic
(uneven hub-driven density) also means a single global `spawn_scale` can be
safe for most seeds and gridlock a minority of them at the *same* grid size
— this is what motivated the gridlock filter work below.

---

## 2. `tune_params.py`'s `growth_ratio` detector has two real bugs

Investigated directly rather than assumed reliable, because it's the basis
for "is this spawn_scale safe" in the existing sweep.

**Bug 1 — window-length dependent.** Re-ran the same seed
(6×6, spawn_scale=0.05, seed=1) at probe lengths of 500/1000/1500/2000 ticks:

| probe ticks | growth_ratio |
|---|---|
| 500  | 1.06 |
| 1000 | 2.25 |
| 1500 | 4.95 |
| 2000 | 3.32 |

The qualitative verdict (locked vs. not) flips depending purely on where the
measurement window closes, because the active-vehicle curve is often still
climbing through the whole run rather than settling into two comparable
halves. **Verified via direct trace:** seed 1's active-vehicle count climbs
from 5 → 1677 continuously across all 2000 ticks, never plateauing — so
"first half vs. second half" is comparing two points on a still-rising curve,
not two stable regimes.

**Bug 2 — false negative on a clear case.** At
5×5/spawn_scale=0.06/seed=2, tail-window active-vehicle count is 1439 (vs.
~114 median across the other 7 seeds in that cell — obviously gridlocked by
inspection) but `growth_ratio` came back 1.29, under the sweep's own 1.4
cutoff. Confirmed by rerunning the sweep's exact logic against this seed
directly — this is a real mislabel in the existing tool, not a one-off.

**Resolution:** built `gridlock_filter.py` using a different signal — see §3.

---

## 3. Gridlock filter — design, and a blind spot found by deliberately trying to break it

**What was built:** `filter_gridlocked_seeds()` in `training/gridlock_filter.py`.
Probes a batch of candidate seeds under the same naive fixed-timer baseline
`tune_params.py` uses, and flags a seed as gridlocked if its late-window
(last 20%) mean active-vehicle count is more than 3× the batch's own median
for that statistic.

**Verified against three sweep cells** with independently-confirmed per-seed
ground truth (6×6/0.05, 4×4/0.07, 5×5/0.06): every actually-gridlocked seed
landed at 3×–16× the batch median; every healthy seed stayed under 1.65×; no
crossover in any tested cell.

**Blind spot found by stress-testing the filter itself:** pushed
spawn_scale to 0.15 at 6×6 (well past the validated safe range) and every
one of 8 seeds gridlocked *together* (tail_active ~1500–1900, all within
~1.2× of each other). The relative/median-based check reported **zero**
outliers, because nothing stood out from an already-bad pack — a
purely-relative check cannot detect uniform badness by construction.

**Fix:** added `ABSOLUTE_TAIL_ACTIVE_PER_INTERSECTION_CEILING`, a second,
absolute check on the batch's own median (normalized per intersection).
Calibrated against tail_active measured at `spawn_scale_lookup.py`'s
validated safe points across grid sizes 3–6 (observed healthy range
~38–95/cell there; ceiling set at 8.0/intersection with headroom above that).
**First implementation of this fix was itself buggy** — see §4.

**Third finding, not a bug — a real property of the system:** re-ran the
filter against 50-seed pools (vs. the original 8-seed sweep) at each grid
size's "safe" spawn_scale from `spawn_scale_lookup.py`, and found nonzero
residual gridlock at 3 of 4 sizes even at the "safe" value:

| grid | safe spawn_scale | gridlock rate (of 50 seeds) |
|---|---|---|
| 3×3 | 0.10 | 12% |
| 4×4 | 0.06 | 4% |
| 5×5 | 0.04 | 0% |
| 6×6 | 0.04 | 8% |

Not a flaw in `spawn_scale_lookup.py` — its job was finding a spawn_scale
where gridlock is *rare* across an 8-seed validation sample, not impossible.
This is the reason a standing per-pool filter is needed in addition to a
good global spawn_scale, rather than treating spawn_scale tuning alone as
sufficient.

---

## 4. Gridlock filter's absolute-ceiling backstop — first version was backwards

**Bug:** the first implementation gated the absolute-ceiling check behind
`healthy_fraction < min_healthy_fraction` (i.e., only check the absolute
ceiling if the relative check already found most seeds unhealthy). Tested
directly against the uniform-gridlock case (spawn_scale=0.15/6×6, all 8
seeds gridlocked together) and the backstop **did not fire** — because
uniform gridlock produces a *high* healthy_fraction (1.0), not a low one:
"healthy" in the relative check only means "not an outlier relative to
peers," which uniform bad seeds trivially satisfy.

**Fix:** removed the `healthy_fraction` gate; the absolute ceiling check now
runs unconditionally against the batch median, regardless of how many seeds
passed the relative check. Re-verified: uniform-gridlock case now correctly
raises `ValueError`; all three previously-validated mixed-outlier cells and
all four known-safe operating points still pass with zero false raises.

---

## 5. `gym_env.py` / `train.py` wiring

**Change, not a bug fix:** `gym_env.py` now filters its seed pool through
`filter_gridlocked_seeds()` on construction (`filter_gridlock=True` by
default), skipped automatically for pools under 4 seeds (not enough for a
meaningful median — the single-city debug pool `seed_pool=[42]` hits this
path and is unaffected).

**Change:** `train.py` switched from `seed_pool=[42]` (single-city, used to
validate the pipeline mechanically) to a real 100-seed candidate pool,
pre-filtered once in `train.py` itself
(`filter_gridlock=False` passed to `TrafficGymEnv` to avoid re-probing an
already-filtered pool a second time). Smoke-tested end-to-end (filter → env
→ PPO, small timestep budget) against the compiled module to confirm the
pipeline runs before handing off — not just read, actually executed.

---

## 6. Trend-check heuristic (`_print_trend_verdict`) — three bugs found across two real training runs

Built to answer "should I extend training" from the reward curve
automatically. Went through three broken versions before it held up; two of
the three were caught by testing against real uploaded training data, not
synthetic cases alone.

### v1 bug — tail-share-only comparison is blind to peak-then-decline

**Design:** compared the rolling mean's improvement in the final 10% of
episodes against total improvement across the run.

**Bug, found against the first real training run (peak ep910, final
ep1000):** the run's rolling mean peaked at ep910 (≈-5433) then declined to
≈-6651 by ep1000 — a genuine ~1200-point regression. v1 compared only the
two endpoints of the tail window (`rolling[-1]` vs. `rolling[tail_start]`),
both of which sat below the peak, so the comparison looked "flat" (-10%
tail share) and printed **"flattening — consistent with convergence."**
This was actively wrong: the run had regressed, not converged.

**Also caught, via synthetic test:** a pure linear climb has ~10% of its
total improvement in its final 10% of episodes by construction (even
spread), so a naive threshold above that baseline (originally 15%) would
misclassify a textbook still-climbing curve as converged.

**Fix (v2):** track the run's best-seen (peak) rolling-mean value and where
it occurred, in addition to the tail-share check. Added a dedicated
"peaked-then-regressed" verdict, distinct from both "still trending" and
"flattening."

### v2 bug — peak-position-fraction gate excluded a real regression

**Design:** flagged a regression only if the peak occurred in the first 80%
of the run (`peak_position_frac < 0.8`).

**Bug, found against the same first real run:** peak was at 91% through the
run (ep910/1000) — past the 80% cutoff — so the regression branch never
fired even though the regression (1218 points, 10% of total climb) was
real and independently confirmed via the SB3 log's own `ep_rew_mean` and the
script's own last-20-episode mean, both of which agreed the tail was worse
than mid-training.

**Fix (v3):** replaced the position-fraction gate with an absolute
episodes-after-peak count (`≥ 5% of total episodes`), reasoning that 90
episodes of room-to-recover not being used is meaningful regardless of what
percentage of the run that represents.

### v3 bug — the episode-count gate itself excluded a second real regression

**Design:** required at least 5% of total episodes (50 of 1000) after the
peak before flagging a regression.

**Bug, found against a second real training run:** peak at ep955/1000 — only
45 episodes after the peak, five short of the 50-episode gate. The printed
verdict said "flattening." Recomputed independently from the raw CSV: the
regression was real (18.5% of total climb, `-5172 → -5621`) and **sustained**
— traced the actual per-episode values from peak to end and confirmed a
steady decline that never recovers (ep955: -5172, ep970: -5641, ep985:
-5712, ep1000: -5621), not one noisy last point. The count gate suppressed a
real, confirmed regression for landing 5 episodes short of an arbitrary
cutoff.

**Fix (v4, current):** replaced the episode-count gate with a sustain check:
average the last `min(20, episodes-after-peak)` rolling-mean values and
require *both* that tail average and the final point to sit meaningfully
(>5% of total climb) below peak. This rejects a peak that's genuinely just
the last 1–2 noisy points (tail average would still sit near peak) while
catching a real sustained decline regardless of exactly how many episodes
it's had to run. If fewer than 3 episodes exist after the peak, the function
explicitly declines to judge sustain and falls through to the tail-share
check instead, rather than guessing.

**Verification:** re-ran a 9-case regression suite (both real uploaded runs,
plus synthetic: linear climb, old single-city curve, flat, too-short,
S-curve, noisy-flat, deliberate peak-then-drop, and peak-1-episode-from-end)
against the current version — all 9 now classify correctly.

### Standing caveat

This is a coarse heuristic (no confidence interval, no statistical test
against a no-improvement null), explicitly documented in the code as a
prompt to go look at the PNG with an informed eye, not a substitute for
looking. Two consecutive real training runs peaking mid-to-late and then
declining is now a *pattern*, not a single anomaly — worth investigating the
cause (hyperparameters vs. environment-specific instability) rather than
continuing to treat each run as independent. `explained_variance ≈ 0`
throughout both runs' logs is a candidate lead, not yet confirmed as the
cause.

---

## Open items (not yet investigated)

- **Why the policy regresses late in training**, across two consecutive
  runs — hyperparameters (learning_rate, batch_size) vs. something
  environment-specific haven't been distinguished yet.
- **Fixed-timer baseline** — not yet implemented. Needed before any
  "beats baseline" claim; must run on the same gridlock-filtered seed pool
  and report the same underlying metrics (wait time, stall count), not just
  composite reward.
- **Checkpoint re-evaluation** — re-running the peak checkpoint (e.g. ep910
  or ep955) deterministically against a held-out seed batch, to confirm the
  peak wasn't a one-episode fluke the rolling-mean window happened to catch.
  Discussed, not yet built.