"""
Sweep spawn_scale and grid size under a naive fixed-timer policy to find a
parameter range where traffic is neither empty (no learning signal) nor
permanently gridlocked (also no learning signal, and it would make the
"agent clears congestion" demo impossible since nothing can clear it).

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
    grids = [(3, 3), (4, 4), (6, 6)]
    spawn_scales = [0.02, 0.04, 0.06, 0.08, 0.12]

    print(f"{'grid':<8}{'spawn_scale':<13}{'avg_active':<12}{'completed':<11}{'stalls':<9}{'growth':<9}verdict")
    for grid_w, grid_h in grids:
        prev_completed = None
        for ss in spawn_scales:
            r = run_once(grid_w, grid_h, ss)
            # Real gridlock signature: throughput should rise monotonically
            # with spawn rate in a healthy network. A DROP in completions
            # despite a higher spawn rate means the network has collapsed --
            # this catches saturate-immediately cases the growth-ratio
            # heuristic misses (it only flags "still getting worse", not
            # "already maxed out by the time we started measuring").
            throughput_collapsed = prev_completed is not None and r["completed"] < prev_completed
            if throughput_collapsed:
                r["verdict"] = "GRIDLOCKED (throughput collapsed)"
            prev_completed = r["completed"]
            print(
                f"{r['grid']:<8}{r['spawn_scale']:<13}{r['avg_active']:<12.1f}"
                f"{r['completed']:<11}{r['stalls']:<9}{r['growth_ratio']:<9.2f}{r['verdict']}"
            )