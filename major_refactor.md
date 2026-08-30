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