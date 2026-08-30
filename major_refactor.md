# Realism Refactor — Plan

Written before touching code so there's a record of *why*, not just *what*.
Update this doc as things change; don't let it drift from what's actually
implemented.

---

## 0. The one design call I'm making without asking first — flagging it clearly

You asked for the agent to "know how many cars are on the road, their
destination (varied for each car), and the same routes they take." Read
literally — feeding the RL policy each vehicle's individual destination/route
— I'm not doing that, and here's why:

1. **It would break the fixed-width observation design.** `OBS_PER_INTERSECTION`
   is a constant specifically so one shared policy can generalize across any
   city size (README's core claim, and the reason stage-2 shared-policy is on
   the roadmap). Per-vehicle data is variable-length (0 to dozens of cars per
   road) — there's no way to feed it into a fixed-size per-intersection vector
   without inventing a padding/aggregation scheme, at which point you're back
   to aggregation anyway.
2. **It wouldn't help the decision the agent actually makes.** The agent's
   only action is `{NS-green, EW-green}` per intersection (confirmed in
   `gym_env.py` — we agreed last turn the agent doesn't route). A light-timing
   policy needs to know *how much pressure* is building on each approach, not
   *where each individual car is ultimately headed*. Destination-level detail
   is exactly the information a routing agent would need — which this agent
   isn't.

**What I'm doing instead:** richer *aggregate* per-road state — live vehicle
counts split into "queued at the light" vs. "still driving toward it" (new,
see §2), plus a downstream-blockage signal. This is what actually answers
"does it deal with congestion from multiple cars on the same route" — it's
just expressed as counts and pressure, not individual car identities. If you
want literal per-vehicle awareness for some other reason (e.g. a future
routing-capable agent), that's a different, bigger project — say so and we
scope it separately rather than bolt it onto the light-timing agent.

---

## 1. The realism gap that actually matters most

Re-reading `sim.rs` closely: **roads currently have no travel time.**
`road.length` exists and feeds the router's cost function, but nothing in
`step()` uses it to make a trip actually take time. A vehicle moves from one
road's queue directly into the next road's queue in exactly one tick,
whenever the light is green and the next road has queue room. So "the road"
is really just a zero-length holding pen at the intersection — a 220-unit
arterial and an 80-unit side street take identically long to cross.

This undermines the thing you're asking about directly: **congestion from
"multiple cars on the same route" is currently only expressed as queue depth
at the intersection**, capped by `capacity`. There's no notion of a road
being busy-but-flowing (cars physically spread out along it, still moving)
versus busy-and-jammed (backed up at the light). Every road is either
"has room" or "full," with nothing in between.

This is also why the README's "longer flowing route beats shorter congested
route" story is currently thin: without travel time, a "longer route" only
costs more in the router's *planning* cost function — it's never actually
slower to *drive*, and it can't be *faster* by avoiding a queue, because
there's no queue to avoid until you're already at the light.

**This is the headline fix, Phase 1.**

---

## 2. Phase 1 — core dynamics realism (Rust, `core/src/`)

### 2a. Real travel time per road
- New `Road` fields: `travel_ticks: u32` (derived from `length` and class —
  arterials are faster per unit length, not just higher-capacity) and
  `transit_capacity: usize` (how many vehicles can physically be mid-road at
  once, derived from `length`).
- New `Simulation` state: `transit: Vec<VecDeque<usize>>`, one queue per
  road, holding vehicles that are driving-but-not-yet-at-the-light. Vehicle
  gains a `transit_ticks_left: u32` counter.
- Every tick: decrement transit counters; a vehicle whose counter hits 0
  moves into the existing intersection-queue (`queues[road_idx]`) *if there's
  room* — if the queue's full, it stays in transit, marked `is_stopped`, and
  accrues wait exactly like today's queue-blocked case. This is spillback:
  a jammed light now visibly backs traffic up along the road, not just at
  the light.
- `spawn_vehicles()` and the downstream-capacity check in `step()` switch
  from checking `queues[...].len() < capacity` to checking transit-pool room
  — a road can be "full" without anyone having reached its queue yet.
- Router's cost function switches from `length / capacity` to something
  based on `travel_ticks` (real transit time) with a capacity-based
  congestion-aversion term, since travel time is now real.

### 2b. Saturation flow (multiple vehicles per green per tick)
- Today, a road advances **at most one vehicle per tick**, regardless of
  road class or capacity — a 6-lane arterial and a 1-lane local street clear
  identically. New `Road.saturation_flow: usize` (Arterial 3, Collector 2,
  Local 1) lets wider roads actually drain faster, which is what makes phase
  *timing* (not just phase *existence*) matter more on busy roads — directly
  relevant to whether the light-timing agent's job is meaningfully harder on
  arterials than on side streets.

### 2c. Enriched but still fixed-width observation
- `OBS_PER_INTERSECTION` grows from 6 → 8: add (i) aggregate in-transit
  vehicle count approaching this intersection (cars "about to arrive," not
  just already queued — lets the agent anticipate rather than only react),
  and (ii) a downstream-blockage signal (fraction of this intersection's
  outgoing roads that are near transit-capacity — lets the agent learn that
  turning green here doesn't help if traffic has nowhere to go).
- This is a **breaking change** to the observation shape. Every existing
  checkpoint (`ppo_traffic_stage1_*.zip`, including the run-3 peak at
  250k steps) becomes incompatible and unusable once this lands — training
  starts over. Worth saying explicitly since it closes the loop on the
  "which checkpoint do we use" question from the handoff: none of them,
  going forward.

### What Phase 1 deliberately does NOT touch
- No live/dynamic rerouting (vehicles still plan once at spawn via jittered
  Dijkstra). Real congestion-aware rerouting (recompute cost mid-trip based
  on current queue state) is a genuinely realistic feature but a separate,
  riskier change — proposing it as Phase 2/stretch, not bundling it in.
- No flyovers/grade-separation yet — see Phase 2.
- No change to the RL algorithm/hyperparameters. The peak-then-regress
  investigation from before is a separate, still-open thread — piling a
  dynamics change on top of an already-unstable training setup would make
  it impossible to tell which change caused what if the next run also
  regresses. Worth revisiting *after* Phase 1 lands and the sim is stable,
  not folded in now.

---

## 3. Phase 2 — city scale, density, flyovers (after Phase 1 is verified)

- Bump default `grid_w`/`grid_h` and `num_hubs` for a denser, larger city
  once Phase 1's dynamics are confirmed to behave (bigger city = more ways
  for a dynamics bug to hide).
- Flyovers: new `Road.grade_separated: bool`. A grade-separated road skips
  the phase-allows check entirely (always flows, subject only to its own
  capacity/saturation flow) — models a bypass that doesn't touch a signaled
  intersection. City gen adds a small number of long point-to-point
  connections between high-zone_weight areas, short `travel_ticks` relative
  to distance (that's the whole point of a flyover). Router cost function
  needs no change — flyovers just naturally score better once travel time
  matters (Phase 1 prerequisite).
- Re-tune `spawn_scale_lookup.py`'s safe values — Phase 1 changes what
  "gridlocked" looks like (transit capacity is a new bottleneck that didn't
  exist before), so the existing safe-spawn-scale sweep is stale the moment
  Phase 1 lands, independent of city size changes.

## 4. Phase 3 — fixed-timer baseline
Build once Phase 1's dynamics are the ones being measured against — no
point baselining a sim we're about to change underneath it. Needs: fixed
cycle length + split (probably proportional to `saturation_flow`/class, so
the baseline isn't a strawman), run on the same gridlock-filtered seed pool,
report wait time and stall count separately (not just composite reward), on
the *new* observation-shape sim.

## 5. Phase 4 — retrain the agent, revisit the earlier instability question
Only after 1–3 are solid. The `explained_variance ≈ 0` / peak-then-regress
lead from before is still open and still worth chasing, but on the new sim,
since Phase 1 changes reward magnitude/variance (travel time smooths out
some of the tick-by-tick reward noise, which could independently affect
`explained_variance` — good to know either way, but shouldn't be conflated
with the pre-Phase-1 finding).

---

## 6. Environment split for this work

- I can compile and `cargo test` the Rust core in this sandbox (confirmed
  working — `rustc`/`cargo` installable via apt, lockfile regenerated for
  the older cargo version here). So Phase 1's Rust changes get real
  unit-test coverage before you see them, not just a read-through.
- I can't run `maturin develop`, retrain, or run anything needing
  `stable-baselines3`/`torch` here (disk-constrained, and per your call,
  you're running that side and sending me console output). So: I'll hand
  you compiled-and-tested Rust + the Python-side edits (`gym_env.py` obs
  shape, etc.) written to match, but those Python edits are unverified by
  me — flag anything that looks off when you run it.
- Lockfile note: I regenerated `Cargo.lock` for this sandbox's older cargo
  (1.75). If your local toolchain is newer this should still work, but if
  `cargo build` complains about the lockfile version on your end, delete
  `Cargo.lock` and let it regenerate rather than fighting it.

---

## 7. Status: Phase 1 (§2a–2c) implemented and unit-tested here

`core/src/city.rs` and `core/src/sim.rs` are done. `cargo build` and
`cargo test` both run clean in this sandbox: **19/19 tests pass** (14
original + 5 new, listed below), no warnings.

New tests, each targeting one specific new behavior rather than just
re-running the old suite and hoping:
- `roads_now_have_real_travel_time` — every road has `travel_ticks >= 1`
  and `transit_capacity >= 2`.
- `vehicle_spends_real_time_in_transit_before_queueing` — a road with
  `travel_ticks > 1` has nobody in its queue after only 1 tick.
- `saturation_flow_lets_multiple_vehicles_through_on_wide_roads` — an
  arterial with a deep queue and green light advances >1 vehicle in one
  tick.
- `spillback_blocks_transit_when_queue_is_full` — a vehicle that finishes
  transit into a full queue stays in transit, marked stopped, and
  contributes to wait accounting (doesn't vanish or silently overflow the
  queue).
- `observation_includes_transit_and_downstream_fields` — confirms
  `OBS_PER_INTERSECTION == 8` and that the two new fields actually vary
  (not just zero-padding) and `downstream_blockage` stays in [0, 1].

**Turned out unnecessary:** I assumed Python-side edits would be needed for
the observation-width change (6 → 8). They're not —
`gym_env.py`'s `observation_space` reads `self.sim.obs_per_intersection`
dynamically from the Rust binding rather than hardcoding a width, and
`pybindings/src/lib.rs` needed zero changes (`cargo check` passes clean
against the refactored core as-is). So **no Python files were touched** in
this pass.

**Correction to §6 above:** I said I couldn't verify Python-side edits — turns
out there aren't any to verify, which is the best outcome.

### What you need to do on your end
1. `cd core && cargo test` — should also show 19/19 on your machine.
   If your `Cargo.lock` complains about lockfile version, delete it and
   let cargo regenerate (see §6 above — I regenerated it for this
   sandbox's older cargo, 1.75).
2. `cd pybindings && maturin develop` — rebuild the extension module.
   Confirm with `python3 -c "import traffic_sim; s = traffic_sim.TrafficSim(seed=1); print(s.obs_per_intersection)"`
   → should print `8`.
3. **Every existing checkpoint is now unusable** (obs shape changed from
   48-dim to 64-dim on a 6x6 grid, etc.) — this was flagged in §2c, just
   confirming it's really true before you go looking for the old
   `ppo_traffic_stage1_250000_steps.zip` for anything.
4. **`spawn_scale_lookup.py`'s safe values are now stale.** Transit
   capacity is a new spawn-blocking bottleneck that didn't exist before
   (previously only queue capacity blocked spawning) — the old sweep
   doesn't know about it. Re-run `tune_params.py`'s sweep (or
   `gridlock_filter.py`'s probe) against the rebuilt `.so` before trusting
   `TRAIN_SEED_POOL` filtering again. I did not touch
   `spawn_scale_lookup.py`'s numbers — don't assume they're still safe.
5. Send me console output from a short smoke-test run (a few thousand
   ticks on one seed, checking `metrics()`/`total_stall_count()` behave
   sanely and nothing panics) before a full training run — cheaper to
   catch a dynamics bug there than 300k timesteps in.

### Immediate next step (Phase 2/3, not started)
Once you've confirmed Phase 1 behaves correctly in your environment:
city scale/density/flyovers (§3), then the fixed-timer baseline (§4).
Not implemented yet — waiting on your smoke-test results first, since
building Phase 2 on top of unverified Phase 1 dynamics would make any
bug much harder to isolate.

---

## 8. Status: Phase 2 (§3, city scale/density/flyovers) implemented and unit-tested here

Went ahead per your go-ahead rather than waiting on Phase 1 smoke-test
results from your end first — flagging that explicitly since the plan
above says Phase 2 was gated on that. If your smoke test turns up a Phase 1
dynamics bug, Phase 2 sits on top of it and would need re-checking; nothing
below assumes Phase 1 is wrong, but nothing below re-verifies it either.

### 8a. City scale/density bump
`CityGenParams::default()` changed:

| field | before | after |
|---|---|---|
| `grid_w` / `grid_h` | 6 / 6 | 8 / 8 |
| `num_hubs` | 2 | 3 |

64 intersections instead of 36, three hub clusters instead of two — bigger
and denser per the plan's "once Phase 1's dynamics are confirmed to behave"
framing. Every existing test that cares about a specific grid size already
overrides `grid_w`/`grid_h` explicitly (verified by grepping for it before
changing the default — see below), so this default change alone doesn't
silently affect any existing test's meaning.

**One existing test did need a small update, not because of the size bump:**
`grid_roads_are_bidirectional` asserts an exact road count (24, for a 3×3
grid with shortcuts/pruning disabled) and didn't previously disable
flyovers. Added `num_flyovers: 0` to that test's params so it still isolates
what it says it isolates. This is the only pre-existing test that needed a
change — everything else in the original 19 passed unmodified against the
new defaults.

### 8b. Flyovers
New `CityGenParams` fields: `num_flyovers` (default 2), `flyover_min_zone_weight`
(default 0.55 — an intersection must be reasonably hub-adjacent to be a
flyover endpoint), `flyover_speed` (default 40.0, vs. Arterial's 22.0 —
faster per unit length than even an ordinary arterial, which is the whole
mechanism that lets a flyover's real point-to-point length still win on
`travel_ticks` against a shorter multi-hop signaled route).

**Design decisions made without asking, flagging clearly, same spirit as
§0 above:**

1. **Flyovers connect two already-existing high-zone_weight intersections
   directly** (grid_dist ≥ 3, so it's a genuine bypass and not just a
   redundant local edge) rather than adding new non-grid intersections.
   The plan's language ("long point-to-point connections between
   high-zone_weight areas") reads either way; I picked the simpler
   interpretation that doesn't require inventing new intersection
   placement logic. If you actually want flyover-only interchange nodes
   (not existing signaled intersections), that's a bigger change — say so.
2. **Flyovers are always `RoadClass::Arterial`**, using the same
   capacity/transit_capacity formulas as an ordinary Arterial road at the
   flyover's (much longer) `length`. This was a judgment call to keep a
   flyover's vehicle-holding capacity realistic relative to its physical
   size rather than inventing a new capacity formula — its actual
   advantage over an ordinary arterial is `travel_ticks` (speed), not an
   inflated vehicle count.
3. **`base_intensity` for flyovers is drawn from the same range as ordinary
   Arterial roads** (`arterial_intensity`, default `(0.7, 1.0)`). Flyovers
   don't get spawned onto directly by `spawn_vehicles()` in any special
   way — they're just another road in the network that happens to be
   grade-separated — so this only matters if a vehicle's route happens to
   start on a flyover, which is rare but possible.
4. **Router cost function needed literally no change** — confirmed this
   directly rather than assuming. `shortest_route`'s cost is already
   `travel_ticks + congestion-aversion term`, and a flyover's `travel_ticks`
   is computed the same way (`length / speed`) as any other road; Dijkstra
   naturally prefers it when its real travel time beats the signaled
   alternative. This matches the plan's §3 prediction exactly
   ("Router cost function needs no change").

**What Phase 2 does NOT touch, on purpose:**
- `spawn_scale_lookup.py`'s safe values — see §9 below, this is now DONE
  (was stale/open when this section was first written, resolved after
  handoff via a real interactive session with the actual compiled module).
- `spawn_scale_lookup.py`'s constructor kwargs on `TrafficSim` (num_flyovers,
  flyover_min_zone_weight, flyover_speed) — **not yet exposed** in
  `pybindings/src/lib.rs`'s `#[new]` signature. Right now flyover count/
  placement is only controllable by editing `CityGenParams::default()` in
  Rust, not from Python. Added as a known gap rather than silently doing it,
  since the plan didn't explicitly ask for it and I didn't want to grow the
  FFI surface without flagging it. Trivial to add if you want it — same
  pattern as every other `CityGenParams` field already exposed there.
- `training/tune_params.py` and `training/gridlock_filter.py` — these DID
  need code changes after all, not just re-running: both had grid sizes
  hardcoded as literals in their `__main__` blocks (`tune_params.py`'s
  `grids = [(3,3),(4,4),(5,5),(6,6)]`, `gridlock_filter.py`'s `test_cells`),
  independent of `CityGenParams::default()`. Re-running them unmodified
  (which happened first, on the actual machine, before this was caught)
  only re-swept the old sizes and told us nothing about 8×8. Fixed by
  adding `(8, 8)` to `tune_params.py`'s `grids` list and `(8, 8, 0.03)` /
  `(8, 8, 0.04)` to `gridlock_filter.py`'s `test_cells` — see §9 for the
  actual result. Neither script's core logic changed, only these literals.
- Fixed-timer baseline (§4) — not started, per the plan's own ordering
  (Phase 3 comes after Phase 2 is verified).

**Breaking change to the Python `roads()` getter:** `TrafficSim.roads` rows
grew from 4 fields to 5 — `(from, to, class, capacity, grade_separated)`,
previously `(from, to, class, capacity)`. `training/export_city.py` was
updated to unpack this defensively (falls back to `grade_separated=False`
if it sees a 4-tuple, so it still runs against an un-rebuilt older `.so`).
`city_viewer.html` was updated to draw a `grade_separated` road as a dashed,
slightly thicker line so a flyover reads visually distinct from an ordinary
arterial. Neither of these was smoke-tested against the real compiled
module here (same sandbox limitation as Phase 1 — no `maturin develop`) —
flag anything that looks visually off once you rebuild.

### 8c. Tests added (6 new, all passing here)

- `flyovers_are_grade_separated_and_arterial` — every flyover road is
  `RoadClass::Arterial` and has `length` exceeding a single grid hop's max
  (80.0..220.0), confirming it's really point-to-point rather than
  accidentally reusing the short grid-hop length formula.
- `flyover_endpoints_are_high_zone_weight` — every flyover's `from`/`to`
  intersection meets `flyover_min_zone_weight`.
- `zero_flyovers_produces_no_grade_separated_roads` — `num_flyovers: 0`
  is a real off-switch, not just a smaller probability.
- `flyover_generation_is_deterministic_per_seed` — same seed produces
  identical flyover `(from, to)` pairs across two independent generations
  (mirrors the existing `same_seed_same_city` guarantee, extended to the
  new feature).
- `tiny_grid_with_no_eligible_intersections_skips_flyovers_gracefully` —
  an unreachable `flyover_min_zone_weight` (1.5, above the 0.0..=1.0
  clamp) skips flyover generation cleanly rather than panicking or
  looping forever trying to find a nonexistent eligible intersection.
- `grade_separated_road_ignores_destination_phase` (in `sim.rs`) — the one
  test that actually exercises the behavioral point of grade separation
  end-to-end: manually queues a vehicle on a flyover, forces the
  destination intersection's phase to the *opposite* of what the flyover's
  `PhaseGroup` would need if it were an ordinary signaled road, steps once,
  and confirms the vehicle advances anyway.

`core` test count: 19 → 25. `cargo test` run clean in this sandbox, no
warnings.

### What you need to do on your end (Phase 2)

1. `cd core && cargo test` — should show 25/25 (19 original Phase-1 tests +
   6 new Phase-2 tests) on your machine, same lockfile note as §6 applies
   if your local cargo differs from this sandbox's 1.75.
2. `cd pybindings && maturin develop` — rebuild the extension module.
   Confirm the new `roads()` shape:
   `python3 -c "import traffic_sim; s = traffic_sim.TrafficSim(seed=1); s.reset(1); print(s.roads[0])"`
   → should print a 5-tuple ending in `True`/`False`, not a 4-tuple.
3. **Re-run the real spawn_scale sweep at the new 8×8/3-hub default** —
   `training/tune_params.py` and `training/gridlock_filter.py`, not the
   quick 5-seed probe described in §8b above. The old
   `spawn_scale_lookup.py` table only covers 3×3 through 6×6 and is now
   both out-of-range (new default is 8×8) and measured under the *old*
   (smaller, 2-hub) city-gen defaults for the sizes it does cover — don't
   assume interpolating/extrapolating it is safe.
4. **Decide whether you want flyover knobs exposed from Python.** Right
   now `num_flyovers`/`flyover_min_zone_weight`/`flyover_speed` only exist
   in Rust's `CityGenParams::default()` — say so if you want them added to
   `TrafficSim`'s constructor kwargs (small change, same pattern as every
   other exposed field).
5. Send console output from a short smoke-test run (same ask as Phase 1's
   §6 point 5) — a few thousand ticks at the new default size, confirming
   `metrics()`/`total_stall_count()` behave sanely with flyovers in the mix
   and nothing panics — before trusting this for a real training run.

---

## 9. Status: 8×8 spawn_scale re-tune — DONE (real hardware, real compiled module)

Point 3 from §8's handoff list is resolved. Run against the actual
`maturin develop`-built `.so` on your Fedora machine, not a sandbox
approximation — this is real data, not a probe.

**First attempt hit a real bug, caught before trusting the result:**
running `tune_params.py`/`gridlock_filter.py` unmodified after the Phase 2
default bump produced output that *looked* like a re-tune but wasn't —
both scripts hardcode their grid sizes as Python literals in their
`__main__` blocks (`tune_params.py`'s `grids = [...]`, `gridlock_filter.py`'s
`test_cells = [...]`), completely independent of
`CityGenParams::default()`. The first run's output was byte-for-byte the
same 3×3–6×6 cells as the original pre-Phase-2 sweep — it never touched
8×8 at all. Caught by noticing the grid-size column in the output didn't
include 8×8, not by any error or crash. Fixed by adding `(8, 8)` to
`tune_params.py`'s `grids` list and `(8, 8, 0.03)` / `(8, 8, 0.04)` to
`gridlock_filter.py`'s `test_cells` — see the diff for both files. No
change to either script's actual logic.

**`tune_params.py`'s 8×8 row, re-run correctly this time:**

| spawn_scale | mean_active | worst_active | gridlock_rate | verdict |
|---|---|---|---|---|
| 0.01 | 53.3  | 65.6  | 0.00 | ok |
| 0.02 | 108.8 | 134.2 | 0.00 | ok |
| 0.03 | 164.2 | 203.7 | 0.00 | ok |
| 0.04 | 219.9 | 270.6 | 0.00 | ok |
| 0.05 | 274.0 | 339.1 | 0.00 | ok |
| 0.06 | 328.2 | 406.6 | 0.00 | ok |
| 0.07 | 386.4 | 475.6 | 0.00 | ok |
| 0.08 | 444.0 | 549.8 | 0.00 | ok |
| 0.10 | 568.1 | 721.4 | 0.00 | ok |

**Important: this table alone does NOT mean 0.10 is safe at 8×8.**
`gridlock_rate=0.00` all the way up the column is exactly what
`errors&findings.md` §2 already found this detector (`growth_ratio`) gets
wrong — it has a confirmed false negative (missed an obviously-gridlocked
5×5/0.06/seed=2 case, ratio 1.29 vs. the 1.4 cutoff, while that seed's own
tail-window active count was ~12× the other seeds' median). A flat "ok"
column here is consistent with "genuinely fine at every value up to 0.1"
and *also* consistent with "the detector isn't firing," and this table
alone can't distinguish the two. Treating this table as sufficient on its
own would repeat the exact mistake `gridlock_filter.py` was built to catch.

**So `gridlock_filter.py`'s median-ratio method (the one actually trusted
in this codebase) was run against two 8×8 candidates instead of trusting
the table above** — 0.03 and 0.04, bracketing a rough linear-scaling guess
from 6×6's already-validated 0.04 value adjusted for 64 vs. 36
intersections. Both came back clean, 8 seeds each:

- **8×8@0.03** — `predicted_gridlocked=[]`, ratios 0.70–1.14 across all 8
  seeds, tail_active range 122.3–199.9.
- **8×8@0.04** — `predicted_gridlocked=[]`, ratios 0.65–1.22 across all 8
  seeds, tail_active range 148.4–278.2.

Both clean, no crossover, same separation pattern as the three
pre-existing validated cells (6×6@0.05, 4×4@0.07, 5×5@0.06) in the same
run.

**Chose 0.04 over 0.03.** Reasoning: 0.04 matches 6×6's already-validated
value exactly (rather than introducing a fifth distinct number into the
table), and it gives noticeably more traffic density (mean_active ~220 vs
~164) with zero gridlock cost — more for an RL agent to actually learn
signal-timing decisions from. This is a judgment call between two
equally-clean options, not something the data alone forced; 0.03 would
also have been defensible if you'd rather stay conservative.

**Wiring:**
- `spawn_scale_lookup.py`'s `_SAFE_SPAWN_SCALE_POINTS` table now has a
  fifth point: `(8, 0.04)`. Docstring updated with the new mean_active
  figure (219.9, from the table above) and the validation method actually
  used (gridlock_filter.py, not tune_params.py's verdict column).
- **New interpolation gap, flagged in the module itself, not hidden:**
  `safe_spawn_scale(7)` now interpolates between `(6, 0.04)` and
  `(8, 0.04)` — both 0.04, so it returns 0.04 for grid_side=7. This is a
  coincidence of both neighbors landing on the same plateau value, **not**
  evidence that 7×7 itself is safe — 7×7 has never been swept. Don't treat
  a 7×7 result from this function as validated; run the real sweep at 7×7
  directly if you ever actually need that size.
- Sanity-check block in `spawn_scale_lookup.py`'s `__main__` extended with
  the new `(8, 0.04)` point and the `(7, 0.04)` interpolation case; re-run
  locally and all checks pass (`python3 spawn_scale_lookup.py`, pure
  Python, no Rust/compiled-module dependency, ran directly in this
  sandbox to confirm the interpolation logic itself before handoff).

**Standing caveat, carried over from `progress_so_far.md`'s own earlier
finding about the OTHER rows in this same table:** this 8×8 point is
validated at 8 seeds, matching this table's existing entries — but
`progress_so_far.md` already documented that a later 50-seed re-check of
those same "safe" values found nonzero residual gridlock (0–12%) the
original 8-seed sweep missed entirely. This 8×8 point carries that
identical, un-eliminated risk. If this spawn_scale is going into a real
training run (not just further exploration), the same 50-seed check that
was eventually done for the older sizes is worth doing here too before
fully trusting it — not done as part of this pass.

### What's left before Phase 3 (fixed-timer baseline)

Two items outstanding, in order — Phase 3 shouldn't start until at least
the first is done, since building a baseline against an unconfirmed
spawn_scale would bake a bad measurement into the one thing the baseline
exists to make trustworthy.

1. **Smoke test at the confirmed spawn_scale=0.04** — point 5 from §8's
   list, never actually done (§9 only covers the tuning search itself, not
   a confirmation run at the value that search landed on). A few thousand
   ticks against the real compiled module, checking `metrics()` /
   `total_stall_count()` behave sanely and nothing panics, e.g.:

   ```python
   import traffic_sim
   sim = traffic_sim.TrafficSim(seed=1, spawn_scale=0.04)
   sim.reset(1)
   n = sim.num_intersections
   for t in range(3000):
       actions = [0 if (t // 15) % 2 == 0 else 1] * n
       obs, reward, done = sim.step(actions)
       if done:
           break
   active, completed, wait, stalls = sim.metrics()
   print("active", active, "total_stalls", sim.total_stall_count())
   ```

   Worth running across a few seeds (1, 2, 3, 42, 101 — same set used
   throughout this doc), not just seed=1, since seed-to-seed variance is
   exactly what motivated the gridlock filter in the first place.

2. **Optional but recommended: 50-seed residual-gridlock re-check at
   8×8@0.04.** Same method `gridlock_filter.py` already runs (§9), just
   with `seeds = list(range(1, 51))` instead of `range(1, 9)` for the
   8×8@0.04 cell specifically. This is the exact check that caught real
   problems before — `progress_so_far.md`'s own finding (§9's standing
   caveat) is that 8-seed sweeps at "safe" spawn_scale values later turned
   up 0–12% residual gridlock once checked at 50 seeds. Skipping this
   means Phase 3's baseline (and any training after it) runs on a value
   that's only cleared the same bar this codebase has already shown isn't
   always sufficient — not a blocker, but a known, named risk if skipped
   rather than an unknown one.

Point 4 from §8's list (flyover Python constructor kwargs) is unrelated to
either of these and remains open on its own, still your call.

### Immediate next step (Phase 3, not started)
Fixed-timer baseline (§4), gated on item 1 above (smoke test at the
confirmed spawn_scale). Item 2 (50-seed check) is a judgment call — do it
first if you want the baseline built on a value with real confidence
behind it, or proceed to Phase 3 accepting the named risk and circle back
if training later shows symptoms (e.g. a seed pool with hidden gridlocked
members, the same failure mode `gridlock_filter.py` exists to prevent).