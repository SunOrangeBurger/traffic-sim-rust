# Adaptive Traffic Flow — Progress Log

Tracks what's actually built and verified against the original planning doc
(`README(8).md`). This is a working log, not the pitch doc — it says what's
done, what's shaky, and what's still open, honestly.

## Repo layout

```
traffic-sim-rust/
├── Cargo.toml                # workspace root
├── core/                     # pure Rust: city gen + sim, NO pyo3 dependency
│   ├── Cargo.toml
│   └── src/
│       ├── city.rs           # procedural city generator
│       ├── sim.rs            # tick-based traffic sim
│       └── lib.rs
├── pybindings/                # thin PyO3 wrapper, builds via maturin
│   ├── Cargo.toml
│   ├── .venv/                 # Python 3.12 venv (NOT 3.14 — see Gotchas)
│   └── src/lib.rs
└── training/                  # everything that imports the compiled module
    ├── tune_params.py         # sim parameter sweep (no RL involved)
    ├── gym_env.py              # Gymnasium wrapper, stage-1 architecture
    └── train.py                 # SB3 PPO training script
```

Core and bindings are a deliberate split: `core` has zero Python
dependencies and is tested with plain `cargo test`, so the simulation logic
can be verified without ever touching Python/maturin. `pybindings` is a thin
translation layer only.

## What's built and verified

### 1. Procedural city generator (`core/src/city.rs`)
Grid graph of intersections + bidirectional roads, seeded deterministically
(same seed → identical city, verified in tests and by two independent
machines producing byte-identical sweep output — see Tuning below). Each
road gets a random arterial/side classification, capacity, and
`base_intensity` (spawn-rate weight), so the city has uneven traffic density
the way a real one does. A small number of non-grid "shortcut" roads are
added for route diversity.

**Status:** done, 3 passing unit tests (`same_seed_same_city`,
`different_seed_likely_different_city`, `grid_roads_are_bidirectional`).

### 2. Tick-based traffic simulation (`core/src/sim.rs`)
Gym-shaped `reset(seed) -> obs` / `step(actions) -> (obs, reward, done)`.
Vehicles queue on roads, advance one-per-tick when their approach's signal
phase is green and the downstream road has capacity. Deliberately crude
physics (no continuous position/speed) per the original plan's "keep the
physics crude" guidance — the point is queueing + stall dynamics, not a
physics sim.

**Key design points:**
- **Slab arena for vehicles** (`Vec<Option<Vehicle>>` + free-list), not a
  plain `Vec` with `retain`. Caught this as a real bug during development —
  pruning completed vehicles from a plain Vec shifts every later index,
  silently corrupting any queue still referencing them. Regression-tested
  (`queue_indices_stay_valid_after_completions`,
  `completed_vehicle_slots_are_reused_not_leaked`).
- **Stall tracking is a first-class signal**, not inferred from queue
  membership: a vehicle's `is_stopped` flag flips on the moving→stopped
  transition, and `stall_count` increments on stopped→moving. This is what
  lets the reward function distinguish "queued briefly at a green light"
  from "came to a full stop" — the mechanism the original plan needs to make
  "longer flowing route beats shorter stop-start route" learnable.
- **`SimConfig`** separates dynamics knobs (`spawn_scale`, `max_ticks`) from
  a pure reward-shaping knob (`stall_penalty`) — the latter does not affect
  simulation dynamics at all, only how the reward weights a stall vs. a
  wait-tick. Exposed as constructor args on the Python side.
- **Fixed-width per-intersection observation** (6 floats: NS/EW queue
  length, NS/EW wait time, current phase, local intensity), independent of
  city size. This is the actual precondition for cross-city-size
  generalization later — see Known Gaps.

**Status:** done, 8 passing unit tests total (`core/`). Verified
independently on two machines (sandbox + local Fedora dev machine) producing
byte-identical output from the same seeds — strong evidence the
seeded-RNG determinism holds across platforms.

### 3. PyO3 bindings (`pybindings/src/lib.rs`)
`TrafficSim` class exposing `reset`, `step`, `metrics`, `total_stall_count`,
and constructor kwargs for every `CityGenParams`/`SimConfig` field
(`grid_w`, `grid_h`, `max_ticks`, `extra_road_prob`, `arterial_prob`,
`spawn_scale`, `stall_penalty`). Builds via `maturin develop --release`.

**Status:** done, builds clean (one harmless lint warning from an older
pyo3 macro pattern — not a bug, ignorable).

### 4. Parameter tuning (`training/tune_params.py`)
Swept `spawn_scale` × grid size under a naive fixed-timer baseline (global
synchronized phase switch every 15 ticks) to find a congestion range that's
neither empty (no learning signal) nor permanently gridlocked (nothing for
an agent to clear). Caught and fixed a real flaw in the first version of
this script: a "growth ratio" heuristic that only detects congestion *still
rising*, missing cities that gridlock immediately and then plateau at
saturation. Fixed by also flagging non-monotonic throughput vs. spawn rate
(the actual signature of network collapse).

**Result:** `spawn_scale = 0.06` (the default) is safely inside the healthy
zone for every tested grid size (3×3 through 6×6). There's a sharp capacity
cliff for 4×4 between 0.06 (healthy) and 0.08 (fully gridlocked, throughput
collapses from 2458 completions down to 2086 despite more spawns) — evidence
this network doesn't degrade gracefully, so don't assume linear headroom
between untested spawn_scale values.

### 5. Stage-1 SB3 PPO training (`training/gym_env.py`, `training/train.py`)
**Explicitly not the final architecture** — this bakes one fixed grid size
into the observation/action space shape, so "generalization" here means
unseen seeds at the *same* size, not unseen city shapes. Built this way on
purpose, staged before the real shared-policy architecture, to validate the
Rust→PyO3→Gymnasium→SB3 chain works mechanically before adding the
complexity of a custom multi-agent-style VecEnv.

**Diagnostic finding:** training against a random city every episode looked
flat (no visible improvement) at 20k timesteps. Isolated this by overfitting
to a single fixed city (`seed_pool=[42]`) — reward improved consistently
from ~-3500 to ~-3000 over 65 episodes on both the sandbox run and an
independent run on the local dev machine (cleaner trend on the second run:
-3380 → -3000). Conclusion: **the pipeline is not broken** — training on a
new random city every episode is simply a harder problem than 20k timesteps
can show visible progress on, exactly the risk the original plan's own risk
table anticipated ("RL doesn't converge / training looks flat → start
single [city] to validate pipeline before scaling").

**Status:** in progress. Currently re-running the single-city setup with
`device="cpu"` (SB3 correctly warned that GPU is inefficient for a small
MLP policy — more overhead than benefit) and a real timestep budget, to see
genuine convergence before scaling to a real multi-city seed pool.

## Known gaps / design debt (tracked on purpose, not accidental)

- **Fixed-size Gym env ≠ the README's actual generalization claim.** The
  planning doc explicitly calls for a shared policy across intersections
  (fixed-size *local* observation → action, applied identically regardless
  of city shape), which is what would let one trained policy handle a 3×3
  city and a 6×6 city with no retraining. The current stage-1 wrapper
  doesn't do this — it's a deliberate staging choice, not a mistake, but it
  needs a follow-up refactor (custom VecEnv where each intersection is a
  parallel "sub-environment" sharing one city's simulation state) before the
  "generalizes to unseen layouts" claim in the pitch is actually true.
- **`explained_variance` stays near zero** through all training runs so far
  (single-city included), even as reward improves. Not necessarily broken —
  likely just early/short training plus per-episode stochastic vehicle
  spawning adding real variance beyond what the policy controls — but worth
  watching once training scales up; may need `VecNormalize` (reward/obs
  normalization) if it doesn't improve with more steps.
- **`stall_penalty` hasn't been tuned at all** — it's a reward-shaping knob
  only (doesn't touch dynamics), deliberately deferred until there's an
  actual training curve to check whether PPO is over/under-weighting stalls
  vs. plain wait time.
- **No minimum-green-time constraint** on the action space — the policy can
  flip a phase every single tick if it wants to. Not unrealistic-looking
  yet in practice, but not physically constrained either; worth deciding
  whether to add before the demo, since unconstrained flapping could look
  like a bug on camera even if the reward math is fine.
- **`MIN_ROUTE_HOPS`/`MAX_ROUTE_HOPS`** (route length 3–8 hops) are still
  hardcoded constants, not swept or exposed. Lower priority than
  `spawn_scale`, but a conscious deferral, not an oversight.

Next steps

In rough priority order given what's been validated so far:

Fixed-timer baseline eval on the same trained city — cheap, and turns the abstract reward improvement into the concrete "X% less wait time, Y% fewer stalls" number the demo actually needs.
Scale to the real seed pool on the current fixed-size env, to see how much longer convergence takes once city layout varies episode-to-episode (expect meaningfully more than 300k steps).
Stage-2 shared-policy refactor — the architecture change that actually delivers cross-city-size generalization, which is the headline differentiator in the original pitch.
Visualization and demo polish (phases 8–9) once there's a policy worth showing off.

## Gridlock filtering (`training/gridlock_filter.py`)

**Problem.** `tune_params.py`'s multi-seed sweep and `--diagnose-hub-variance`
diagnostic already established that individual seeds can gridlock purely
from hub-placement luck, independent of grid size or spawn_scale being
otherwise reasonable (~19x avg_active spread across seeds at a single fixed
6x6/spawn_scale=0.05 cell). A gridlocked seed has no learnable signal —
queues grow unbounded regardless of signal policy, because the network's
throughput ceiling has been exceeded, not because timing is suboptimal.
Left in a training pool, it corrupts the reward signal; drawn live in the
demo, it would make the agent visibly fail to clear congestion on stage —
the one thing the demo script promises judges.

**First finding: the existing `growth_ratio` detector isn't reliable enough
to build a filter on.** Investigated it directly (not just assumed) and
found two real problems:
1. It's window-length dependent — re-running the same seed (6x6/0.05/seed=1)
   at probe lengths of 500/1000/1500/2000 ticks gave growth_ratio =
   1.06 / 2.25 / 4.95 / 3.32. The qualitative verdict (locked vs not) flips
   depending purely on where the measurement window closes, because the
   active-vehicle curve is often still climbing through the whole run
   rather than settling into two comparable halves.
2. It has a false negative on a clear case — 5x5/spawn_scale=0.06/seed=2 has
   a tail-window active-vehicle count of 1439 (vs. ~114 median across the
   other 7 seeds in that cell — obviously gridlocked) but growth_ratio came
   back 1.29, under the 1.4 cutoff used in the original sweep. Confirmed by
   rerunning the sweep's exact logic against this seed directly.

**What was built instead.** `filter_gridlocked_seeds()` probes a batch of
candidate seeds under the same naive fixed-timer baseline tune_params.py
already uses, and compares each seed's late-window (last 20%) mean
active-vehicle count against the *median* of that statistic across the
batch. This separates cleanly: verified against three sweep cells from the
original tuning data, every actually-gridlocked seed lands at 3x–16x the
batch median, every healthy seed stays under 1.65x, with no crossover in
any tested cell. Using the batch's own median (rather than one hardcoded
vehicle-count cutoff) is what makes one threshold work across different
grid sizes without retuning, since capacity scales with network size but a
self-relative ratio doesn't need to.

**Second finding, caught by deliberately trying to break the filter: a
purely relative check has a blind spot.** Pushed spawn_scale to 0.15 at 6x6
(well past the validated safe range) and every one of 8 seeds gridlocked
*together* — the relative check reported zero outliers, because nothing
stood out from an already-bad pack. Added a second, absolute check
(`ABSOLUTE_TAIL_ACTIVE_PER_INTERSECTION_CEILING`) that raises if the
batch's own median tail-activity (normalized per intersection) is itself
too high, regardless of how many seeds "passed" the relative check — a
high pass rate is exactly what uniform gridlock looks like, not evidence
the pool is fine. This is a backstop for a badly-chosen spawn_scale, not a
replacement for `spawn_scale_lookup.py`.

**Third finding: residual gridlock exists even at the "safe" spawn_scale.**
`spawn_scale_lookup.py`'s safe values were tuned against an 8-seed sweep.
Ran the new filter against 50-seed pools at each grid size's safe
spawn_scale and found a non-zero gridlock rate at 3 of 4 sizes:
3x3 → 12%, 4x4 → 4%, 5x5 → 0%, 6x6 → 8%. Not a flaw in
`spawn_scale_lookup.py` (its job was finding a spawn_scale where gridlock
is rare, not impossible, across a small validation sample) — it's the
reason a standing per-pool filter is needed in addition to a good global
spawn_scale, rather than treating spawn_scale tuning alone as sufficient.

**Wiring.** `gym_env.py` now filters its seed pool through
`filter_gridlocked_seeds()` by default (`filter_gridlock=True`), skipped
automatically for pools under 4 seeds (e.g. the single-city debug pool
`seed_pool=[42]`, which doesn't have enough seeds for a meaningful median
anyway). `train.py` now trains against a real 100-seed candidate pool
(previously `seed_pool=[42]`, deliberately overfitting one city to validate
the pipeline mechanically — see the Stage-1 section above) filtered once
before constructing the env, passing `filter_gridlock=False` to
`TrafficGymEnv` to avoid re-probing an already-filtered pool a second time.

**Status:** done and tested against the pretrained module — filter logic
verified against three sweep cells with known per-seed outcomes, the
uniform-gridlock backstop verified by deliberately forcing it to fire, and
the full `train.py` pipeline (filter → env → PPO) smoke-tested end-to-end
(small timestep budget) to confirm it runs. The actual long
multi-city 300k-step convergence run — the thing `progress_so_far.md`
originally deferred pending this fix — hasn't been run yet; that's the
immediate next step, not part of this fix.

**Known limitation carried forward:** the filter is a pool-hygiene step
(reject bad seeds before training), not a simulation fix — the underlying
network-capacity cliff `tune_params.py` found (e.g. 4x4 collapsing between
spawn_scale 0.06 and 0.08) still exists in the sim itself. If Stage-2
broadens the grid-size range trained on, this filter should be re-validated
at those new sizes rather than assumed to generalize, the same way
`spawn_scale_lookup.py` explicitly refuses to extrapolate past its tested
range.