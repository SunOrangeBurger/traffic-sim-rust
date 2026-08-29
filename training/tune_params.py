"""
Sweep spawn_scale and grid size under a naive fixed-timer policy to find a
parameter range where traffic is neither empty (no learning signal) nor
permanently gridlocked (also no learning signal, and it would make the
"agent clears congestion" demo impossible since nothing can clear it).

Averages over multiple seeds per (grid, spawn_scale) cell. Hub cluster
placement is randomized per-city (see core/src/city.rs), so any single seed
can land two hubs adjacently and produce a fluke over-dense city regardless
of grid size -- confirmed via --diagnose-hub-variance, which showed ~19x
avg_active spread across seeds at a FIXED size and spawn_scale. A one-seed
sweep therefore reads seed luck, not the size's actual typical behavior;
this file reports mean and worst-case across seeds instead so the resulting
spawn_scale choice reflects what stage-2 training will actually see across
a random seed pool, not one arbitrary hub roll.

stall_penalty is intentionally NOT swept here: it doesn't affect simulation
dynamics at all, only reward scale. Sweep it later, during training, by
watching whether PPO's reward curve is dominated by one term.
"""
import traffic_sim

FIXED_TIMER_PERIOD = 15  # ticks per phase; a genuinely naive baseline
RUN_TICKS = 2000
WARMUP_FRACTION = 0.2  # ignore the first 20% of ticks (city still filling up)


def fixed_timer_actions(n_intersections, tick, period=FIXED_TIMER_PERIOD):
    phase = 0 if (tick // period) % 2 == 0 else 1
    return [phase] * n_intersections


def run_once(grid_w, grid_h, spawn_scale, seed=42, run_ticks=RUN_TICKS):
    sim = traffic_sim.TrafficSim(
        seed=seed,
        grid_w=grid_w,
        grid_h=grid_h,
        max_ticks=run_ticks,
        spawn_scale=spawn_scale,
    )
    sim.reset(seed)
    n = sim.num_intersections

    warmup_end = int(run_ticks * WARMUP_FRACTION)
    active_history = []
    completed_total = 0
    stall_total_end = 0

    for t in range(run_ticks):
        actions = fixed_timer_actions(n, t)
        obs, reward, done = sim.step(actions)
        active, completed_this_tick, total_wait, stalls_this_tick = sim.metrics()
        completed_total += completed_this_tick
        if t >= warmup_end:
            active_history.append(active)
        if done:
            break

    stall_total_end = sim.total_stall_count()

    # Divergence check: compare mean active-vehicle count in the first vs
    # second half of the post-warmup window. Steady-state queueing networks
    # should hover around a roughly constant level; a growing mean means the
    # arrival rate is outrunning the fixed-timer's departure rate --
    # permanent gridlock, not "congestion the agent can learn to clear."
    mid = len(active_history) // 2
    first_half_mean = sum(active_history[:mid]) / max(mid, 1)
    second_half_mean = sum(active_history[mid:]) / max(len(active_history) - mid, 1)
    growth_ratio = second_half_mean / max(first_half_mean, 1e-6)

    return {
        "grid": f"{grid_w}x{grid_h}",
        "spawn_scale": spawn_scale,
        "avg_active": sum(active_history) / len(active_history),
        "completed": completed_total,
        "stalls": stall_total_end,
        "growth_ratio": growth_ratio,
        "verdict": (
            "GRIDLOCKING" if growth_ratio > 1.4
            else "too sparse" if sum(active_history) / len(active_history) < 2
            else "ok"
        ),
    }


if __name__ == "__main__":
    import sys

    if "--diagnose-hub-variance" in sys.argv:
        # One-off diagnostic: is the 5x5-vs-6x6 non-monotonicity in the main
        # sweep a real size effect, or an artifact of hub_centers being
        # drawn from a single fixed seed (42) per size, so different sizes
        # just happen to get luckier/unluckier hub layouts? Runs several
        # seeds per size at a fixed mid-range spawn_scale and reports the
        # SPREAD across seeds -- if within-size seed variance is as large
        # as the between-size gap seen in the main sweep, that confirms
        # it's noise, not a structural size effect.
        #
        # CONFIRMED (see progress log): within one grid size, avg_active
        # swings ~19x across seeds (57.8 to 1071.2 at 6x6, spawn_scale=0.05)
        # purely from hub-placement luck -- a bigger spread than the
        # apparent between-size gap in the single-seed main sweep. Decision:
        # kept as-is (real road networks have unlucky bottleneck layouts
        # too), so the main sweep below now averages over multiple seeds
        # per cell instead of trusting any single seed.
        DIAG_SPAWN_SCALE = 0.05
        DIAG_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
        print(f"Diagnostic: spawn_scale={DIAG_SPAWN_SCALE}, seeds={DIAG_SEEDS}")
        print(f"{'grid':<8}{'seed':<6}{'avg_active':<12}{'completed':<11}{'growth':<9}verdict")
        for grid_w, grid_h in [(5, 5), (6, 6)]:
            results = []
            for seed in DIAG_SEEDS:
                r = run_once(grid_w, grid_h, DIAG_SPAWN_SCALE, seed=seed)
                results.append(r)
                print(
                    f"{r['grid']:<8}{seed:<6}{r['avg_active']:<12.1f}"
                    f"{r['completed']:<11}{r['growth_ratio']:<9.2f}{r['verdict']}"
                )
            avg_actives = [r["avg_active"] for r in results]
            print(
                f"  -> {grid_w}x{grid_h} avg_active across seeds: "
                f"min={min(avg_actives):.1f} max={max(avg_actives):.1f} "
                f"mean={sum(avg_actives)/len(avg_actives):.1f}"
            )
        sys.exit(0)

    # 5x5 added: stage-2 trains across a size range (3x3-6x6) and needs the
    # midpoint covered too, not just the endpoints, since a shared policy
    # will actually see this size during training.
    grids = [(3, 3), (4, 4), (5, 5), (6, 6)]
    spawn_scales = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    # Hub placement is randomized per-city (see diagnose-hub-variance above),
    # so any single seed can land a fluke dense or fluke sparse layout.
    # Averaging over several seeds per cell is what makes this sweep read
    # the TYPICAL healthy zone rather than one arbitrary hub roll.
    TUNE_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

    print(
        f"{'grid':<8}{'spawn_scale':<13}{'mean_active':<13}{'worst_active':<14}"
        f"{'gridlock_rate':<15}verdict"
    )
    for grid_w, grid_h in grids:
        for ss in spawn_scales:
            per_seed = [run_once(grid_w, grid_h, ss, seed=s) for s in TUNE_SEEDS]
            actives = [r["avg_active"] for r in per_seed]
            # A seed counts as "gridlocked" for this cell using the same
            # signal run_once already computes per-seed: high growth_ratio
            # (still-rising congestion) is the honest per-seed proxy here,
            # since the throughput-collapse comparison needs a previous
            # spawn_scale at the SAME seed, which this seed-averaged view
            # doesn't track -- growth_ratio alone is sufficient to flag the
            # runaway cases seen in the diagnostic (all were growth_ratio
            # 2.5+).
            gridlocked_seeds = sum(1 for r in per_seed if r["growth_ratio"] > 1.4)
            mean_active = sum(actives) / len(actives)
            worst_active = max(actives)
            gridlock_rate = gridlocked_seeds / len(TUNE_SEEDS)
            verdict = (
                "too sparse" if mean_active < 2
                else f"UNSTABLE ({gridlocked_seeds}/{len(TUNE_SEEDS)} seeds gridlocked)" if gridlock_rate > 0
                else "ok"
            )
            print(
                f"{grid_w}x{grid_h:<6}{ss:<13}{mean_active:<13.1f}{worst_active:<14.1f}"
                f"{gridlock_rate:<15.2f}{verdict}"
            )