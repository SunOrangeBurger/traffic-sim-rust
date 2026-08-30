"""
Per-seed gridlock filter.

Problem this solves: `tune_params.py`'s multi-seed sweep already showed that
at a fixed (grid_size, spawn_scale), individual seeds can gridlock purely
from hub-placement luck (--diagnose-hub-variance: ~19x avg_active spread
across seeds at 6x6/spawn_scale=0.05). A gridlocked seed has no learnable
signal -- queues grow without bound regardless of signal policy, because the
network's throughput ceiling has been exceeded, not because timing is bad.
Left in a training or demo seed pool, it corrupts the reward signal during
training and risks a visibly-broken live demo if it's the seed drawn on
stage.

This module rejects such seeds *before* they enter a seed pool, via a cheap
probe run under the same naive fixed-timer baseline tune_params.py already
uses for its sweep.

## Why growth_ratio (tune_params.py's existing signal) isn't used here

`tune_params.py` flags a seed as gridlocked when
`second_half_mean / first_half_mean > 1.4` over the post-warmup window.
Investigated this directly against ground truth and found it unreliable in
two ways:

1. **Window-length dependent.** Re-ran 6x6/spawn_scale=0.05/seed=1 at probe
   lengths of 500/1000/1500/2000 ticks: growth_ratio came back
   1.06 / 2.25 / 4.95 / 3.32 -- swinging by 4x+ depending purely on where the
   window closes, because the active-vehicle curve is still climbing
   through the whole 2000-tick run, not settling into two comparable
   halves. A metric that changes qualitative verdict (locked vs not)
   depending on probe length isn't safe to shorten for a cheap early check.

2. **False negative on a clear case.** At 5x5/spawn_scale=0.06/seed=2, the
   tail-window active-vehicle count is 1439 (vs. a same-cell median of
   ~114 across the other 7 seeds -- obviously gridlocked by inspection) but
   growth_ratio came back 1.29, under the 1.4 cutoff, so the existing sweep
   silently mislabels this seed as healthy. Confirmed by rerunning the
   exact sweep logic from tune_params.py against this seed directly.

## What this module uses instead

Late-window (last 20% of probe ticks) mean active-vehicle count, compared
against the *median* of that same statistic across a batch of candidate
seeds for the same (grid_w, grid_h, spawn_scale) cell. Gridlocked seeds
separate cleanly this way -- verified against three sweep cells pulled from
the original tuning data (6x6/0.05, 4x4/0.07, 5x5/0.06): every gridlocked
seed lands at 3x-16x the batch median, every healthy seed stays under 1.65x,
no crossover in any of the three cells tested. Using the batch's own median
(rather than one hardcoded absolute number) is what makes the same threshold
work across different grid sizes without retuning -- capacity scales with
network size, so an absolute vehicle-count cutoff would need a different
constant per grid size, but a self-relative ratio doesn't.

Tradeoff: this requires probing a batch of candidate seeds together (needs
at least a handful of seeds to get a meaningful median) rather than judging
one seed in isolation. That fits how this is actually used -- filtering a
seed pool before training, not a single ad hoc seed check.

## Known limitation: uniform gridlock defeats the outlier check

The median-relative approach only works when gridlock is the *exception*
across the batch, not the rule -- it looks for seeds that are unusually bad
relative to their peers. Verified this fails exactly as expected: pushed
spawn_scale to 0.15 at 6x6 (well past the safe range in
spawn_scale_lookup.py) and every one of 8 seeds gridlocked together
(tail_active ~1500-1900, all within ~1.2x of each other) -- the filter
reported ZERO seeds as outliers, because none of them stood out from an
already-bad pack. This is not a bug in the outlier logic, it's a real gap
in what a *relative* check alone can catch. `filter_gridlocked_seeds` adds
an absolute floor (`ABSOLUTE_TAIL_ACTIVE_PER_INTERSECTION_CEILING`)
specifically to catch this case: if the batch's own median tail-activity
(normalized per intersection) exceeds that ceiling, it raises rather than
silently returning a falsely-reassuring "all healthy" result -- regardless
of how many seeds passed the relative check, since a high pass rate is
exactly what uniform gridlock produces (see `filter_gridlocked_seeds`'
docstring for why the check can't be gated behind a low pass rate). This
is a backstop, not a replacement for picking spawn_scale sanely in the
first place -- see spawn_scale_lookup.py.
"""
import statistics

import traffic_sim

FIXED_TIMER_PERIOD = 15  # matches tune_params.py's naive baseline exactly
TAIL_FRACTION = 0.2  # look at the last 20% of the probe window
DEFAULT_PROBE_TICKS = 2000  # matches tune_params.py's RUN_TICKS; see note below
OUTLIER_RATIO_THRESHOLD = 3.0  # a seed at >3x the batch median tail-activity is gridlocked


def _fixed_timer_actions(n, tick, period=FIXED_TIMER_PERIOD):
    phase = 0 if (tick // period) % 2 == 0 else 1
    return [phase] * n


def _tail_mean_active(grid_w, grid_h, spawn_scale, seed, probe_ticks):
    """Run one seed under the naive fixed-timer baseline and return the mean
    active-vehicle count over the last TAIL_FRACTION of the run. NOTE: probe
    length matters -- see module docstring point (1). Do not shorten
    probe_ticks below DEFAULT_PROBE_TICKS without re-validating against
    known seeds the way this module's __main__ block does, since the
    active-vehicle curve for a gridlocking seed is often still climbing at
    shorter windows and won't yet show separation from healthy seeds.
    """
    sim = traffic_sim.TrafficSim(
        seed=seed, grid_w=grid_w, grid_h=grid_h,
        max_ticks=probe_ticks, spawn_scale=spawn_scale,
    )
    sim.reset(seed)
    n = sim.num_intersections
    history = []
    for t in range(probe_ticks):
        actions = _fixed_timer_actions(n, t)
        sim.step(actions)
        active, _completed, _wait, _stalls = sim.metrics()
        history.append(active)
    tail = history[int(len(history) * (1 - TAIL_FRACTION)):]
    return sum(tail) / len(tail)


#: Absolute per-intersection tail-activity ceiling used as a backstop when
#: the whole batch is uniformly congested (see module docstring, "Known
#: limitation: uniform gridlock defeats the outlier check"). Calibrated
#: against tail_active measured at spawn_scale_lookup.py's own validated
#: safe operating point across grid sizes 3-6 (observed range ~38-95 per
#: cell there); set with headroom above that observed range rather than
#: tight against it, since some healthy variance is expected and this is a
#: backstop for the uniform-gridlock case, not the primary detector.
ABSOLUTE_TAIL_ACTIVE_PER_INTERSECTION_CEILING = 8.0


def filter_gridlocked_seeds(
    candidate_seeds,
    grid_w,
    grid_h,
    spawn_scale,
    probe_ticks=DEFAULT_PROBE_TICKS,
    outlier_ratio_threshold=OUTLIER_RATIO_THRESHOLD,
    min_healthy_fraction=0.5,
):
    """Given a batch of candidate seeds for one (grid_w, grid_h, spawn_scale)
    cell, probe each and return (healthy_seeds, gridlocked_seeds, diagnostics).

    diagnostics is a dict {seed: {"tail_active": float, "ratio_to_median": float}}
    for logging/inspection -- e.g. to sanity-check a borderline threshold call
    before trusting it for a large seed pool.

    Needs >= 4 candidate seeds to get a median worth comparing against; fewer
    than that and a single unlucky seed could itself skew the median. Raises
    ValueError below that, rather than silently returning a meaningless
    filter result.

    Raises ValueError if the batch's own median tail-activity exceeds
    ABSOLUTE_TAIL_ACTIVE_PER_INTERSECTION_CEILING -- this is the backstop
    for uniform gridlock (see module docstring), where every seed is bad
    *together* and the relative outlier check alone wrongly reports "all
    healthy" because nothing stands out from an already-bad pack (verified:
    at spawn_scale=0.15/6x6, all 8 seeds cluster within ~1.2x of each other
    despite every single one being deep in gridlock, so healthy_fraction
    comes back 1.0 and would silently pass without this check). This is why
    the absolute ceiling is checked against the median directly, not gated
    behind a low healthy_fraction -- uniform gridlock produces a HIGH
    healthy_fraction by construction, since "healthy" here only means
    "not an outlier relative to peers," which uniform bad seeds trivially
    satisfy. `min_healthy_fraction` is accepted for API stability /
    future use but a low value alone (with a low median) does not raise.
    """
    if len(candidate_seeds) < 4:
        raise ValueError(
            f"filter_gridlocked_seeds needs >= 4 candidate seeds to compute a "
            f"meaningful batch median, got {len(candidate_seeds)}"
        )

    n_intersections = grid_w * grid_h
    tail_active = {
        seed: _tail_mean_active(grid_w, grid_h, spawn_scale, seed, probe_ticks)
        for seed in candidate_seeds
    }
    median = statistics.median(tail_active.values())

    healthy, gridlocked = [], []
    diagnostics = {}
    for seed, tail in tail_active.items():
        ratio = tail / max(median, 1e-6)
        diagnostics[seed] = {"tail_active": tail, "ratio_to_median": ratio}
        if ratio > outlier_ratio_threshold:
            gridlocked.append(seed)
        else:
            healthy.append(seed)

    healthy_fraction = len(healthy) / len(candidate_seeds)
    median_per_intersection = median / n_intersections
    if median_per_intersection > ABSOLUTE_TAIL_ACTIVE_PER_INTERSECTION_CEILING:
        raise ValueError(
            f"batch-wide gridlock suspected for grid={grid_w}x{grid_h} "
            f"spawn_scale={spawn_scale}: median tail_active={median:.1f} "
            f"({median_per_intersection:.1f}/intersection, ceiling="
            f"{ABSOLUTE_TAIL_ACTIVE_PER_INTERSECTION_CEILING}) with "
            f"{healthy_fraction:.0%} of the batch passing the relative "
            f"outlier check -- a high pass rate here is exactly what "
            f"uniform gridlock looks like (see module docstring), not "
            f"evidence the pool is fine. spawn_scale is very likely too "
            f"high for this grid size; check spawn_scale_lookup.py."
        )

    return healthy, gridlocked, diagnostics


if __name__ == "__main__":
    # Validate against three cells pulled from the original tune_params.py
    # sweep (README's pasted terminal output) where per-seed gridlock status
    # is independently derivable from tail-window active count. This is a
    # regression check for this module, not a general-purpose tool run.
    print(f"{'cell':<16}{'predicted_gridlocked':<24}note")
    test_cells = [
        (6, 6, 0.05),
        (4, 4, 0.07),
        (5, 5, 0.06),
    ]
    for grid_w, grid_h, ss in test_cells:
        seeds = list(range(1, 9))
        healthy, gridlocked, diag = filter_gridlocked_seeds(seeds, grid_w, grid_h, ss)
        cell_label = f"{grid_w}x{grid_h}@{ss}"
        print(f"{cell_label:<16}{str(sorted(gridlocked)):<24}")
        for seed in sorted(seeds):
            d = diag[seed]
            flag = "GRIDLOCKED" if seed in gridlocked else "ok"
            print(f"    seed={seed} tail_active={d['tail_active']:<8.1f} "
                  f"ratio_to_median={d['ratio_to_median']:<6.2f} {flag}")