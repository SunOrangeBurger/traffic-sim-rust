"""
Stage-1 training: fixed grid size, standard SB3 PPO (no shared-policy
architecture yet -- see gym_env.py's module docstring).

Now trains against a real multi-seed pool (previously overfit to a single
fixed city, seed_pool=[42], to validate the pipeline mechanically -- see
progress_so_far.md for that result). The candidate pool is passed through
gridlock_filter.py first: an unfiltered random-seed pool would include some
fraction of gridlocked cities with no learnable signal (unbounded queues
regardless of policy), which would corrupt the reward signal in a way
that's easy to misdiagnose as "PPO isn't converging."

Forces CPU (SB3 warns MlpPolicy-on-GPU is usually slower, not faster, for
networks this small), adds checkpointing so a long run survives an
interrupt, and saves a reward-curve plot so "is it converging" is a glance
at a PNG instead of scrolling console output.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from gym_env import TrafficGymEnv
from spawn_scale_lookup import safe_spawn_scale
from gridlock_filter import filter_gridlocked_seeds


class EpisodeRewardLogger(BaseCallback):
    """Collects every episode's reward, prints a rolling mean periodically,
    and (on training end) writes the full history to CSV + a PNG plot so you
    can actually see the convergence trend rather than scrolling logs."""

    def __init__(self, log_every=20, out_dir="."):
        super().__init__()
        self.log_every = log_every
        self.out_dir = out_dir
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                if len(self.episode_rewards) % self.log_every == 0:
                    recent = self.episode_rewards[-self.log_every :]
                    print(
                        f"episodes {len(self.episode_rewards) - self.log_every + 1}"
                        f"-{len(self.episode_rewards)}: "
                        f"mean reward = {sum(recent) / len(recent):.1f}"
                    )
        return True

    def _on_training_end(self) -> None:
        if not self.episode_rewards:
            return
        csv_path = os.path.join(self.out_dir, "reward_curve.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])
            for i, r in enumerate(self.episode_rewards, start=1):
                writer.writerow([i, r])
        print(f"Saved {csv_path} ({len(self.episode_rewards)} episodes)")

        # rolling mean over a window so the plot is readable despite
        # per-episode noise from stochastic vehicle spawning/routing
        window = max(1, len(self.episode_rewards) // 50)
        rolling = [
            sum(self.episode_rewards[max(0, i - window) : i + 1])
            / len(self.episode_rewards[max(0, i - window) : i + 1])
            for i in range(len(self.episode_rewards))
        ]
        plt.figure(figsize=(9, 5))
        plt.plot(self.episode_rewards, alpha=0.25, label="per-episode reward")
        plt.plot(rolling, linewidth=2, label=f"rolling mean (window={window})")
        plt.xlabel("episode")
        plt.ylabel("episode reward")
        plt.title("PPO training curve (multi-city, gridlock-filtered seed pool)")
        plt.legend()
        plt.tight_layout()
        png_path = os.path.join(self.out_dir, "reward_curve.png")
        plt.savefig(png_path, dpi=120)
        print(f"Saved {png_path}")

        self._print_trend_verdict(rolling)

    def _print_trend_verdict(self, rolling):
        """Cheap, honest answer to 'should I extend training' without having
        to eyeball the PNG.

        v1 of this check only compared the last data point against a point
        10% back, looking for the tail's SHARE of total improvement. Found a
        real bug in it via an actual run: a policy that peaks mid-training
        and then regresses (rolling mean up to ~-5433 around episode 910,
        back down to ~-6651 by episode 1000) got called "flattening" --
        because comparing two points near a shared trough after a decline
        looks numerically flat/negative-and-small even though real
        degradation happened in between. A single endpoint comparison is
        blind to peak-then-decline entirely.

        v2 tracks the run's best-seen rolling mean and reports where in the
        run it occurred, in addition to the tail-share check. This is what
        actually distinguishes three cases that "still improving vs.
        flattening" alone conflates:
          - genuinely still climbing at the end (peak == last episode)
          - converged/plateaued near the peak (peak recent, tail flat)
          - peaked then regressed (peak well before the end, tail below it)
        The third case needs a different response than "run more steps" --
        it's evidence of late-training instability (e.g. PPO policy
        collapse/overfitting to recent episodes), which more steps of the
        SAME run won't reliably fix and could make worse. Checking
        `explained_variance` in the training log is the natural next step
        there, not just extending total_timesteps.

        Still a coarse heuristic, not a rigorous convergence test -- treat
        it as a prompt to go look at the PNG with an informed eye, not as a
        substitute for looking.
        """
        n = len(rolling)
        if n < 20:
            print(
                "[trend check] fewer than 20 episodes logged -- too short to "
                "say anything meaningful about convergence."
            )
            return

        peak_idx = max(range(n), key=lambda i: rolling[i])
        peak_value = rolling[peak_idx]
        peak_episode = peak_idx + 1  # 1-indexed for human-readable output
        final_value = rolling[-1]
        peak_position_frac = peak_idx / (n - 1) if n > 1 else 1.0

        tail_start = int(n * 0.9)
        total_improvement = final_value - rolling[0]
        tail_improvement = final_value - rolling[tail_start]

        print(
            f"[trend check] rolling mean: start={rolling[0]:.1f}, "
            f"peak={peak_value:.1f} at episode {peak_episode} "
            f"({peak_position_frac:.0%} through the run), "
            f"final={final_value:.1f}."
        )

        # Peak well before the end AND final noticeably below peak: the
        # policy got worse late in training, not just "not yet converged".
        #
        # v2 gated this on a raw episode COUNT after the peak (>= 5% of n).
        # Found a real bug via a second actual run: peak at episode 955/1000
        # (45 episodes after peak, just under the 50-episode gate) showed an
        # 18.5%-of-climb sustained regression -- checked the raw values from
        # peak to end and confirmed it's a real decline that never recovers
        # (-5172 -> -5296 -> -5408 -> ... -> -5621), not one noisy last
        # point. The count gate suppressed a real regression for landing 5
        # episodes short of an arbitrary cutoff.
        #
        # v3 replaces the count gate with a SUSTAIN check: instead of asking
        # "how many episodes came after the peak", ask "is the final value's
        # dip corroborated by the tail window average, not just a single
        # possibly-noisy last point". Average the last min(20, available)
        # episodes and require both that average AND the final point to sit
        # meaningfully below peak -- this rejects a peak that's genuinely
        # just the last 1-2 noisy points (tail average would still be near
        # peak) while catching a real sustained decline regardless of
        # exactly how many episodes it's had to run.
        tail_window = min(20, n - peak_idx - 1) if n - peak_idx - 1 > 0 else 0
        climb_to_peak = peak_value - rolling[0]
        regression_from_peak = peak_value - final_value
        reward_spread = max(rolling) - min(rolling)
        meaningful_climb = climb_to_peak > max(1.0, reward_spread * 0.1)

        if tail_window >= 3:
            tail_avg = sum(rolling[-tail_window:]) / tail_window
            tail_regression = peak_value - tail_avg
            sustained = (
                meaningful_climb
                and regression_from_peak / climb_to_peak > 0.05
                and tail_regression / climb_to_peak > 0.05
            )
        else:
            # Peak is right at (or 1-2 episodes from) the end -- not enough
            # room after it to distinguish "real decline" from "one noisy
            # point at the very tail". Don't flag as regressed; the
            # tail-share check below (which looks at trend over a wider
            # window) is the more appropriate signal here.
            sustained = False

        regressed = sustained

        if regressed:
            print(
                f"[trend check] VERDICT: peaked-then-regressed -- best "
                f"performance was {regression_from_peak:.1f} better than "
                f"the final value, and occurred well before training ended. "
                f"This is NOT the same as 'needs more steps' -- simply "
                f"extending total_timesteps continues from the regressed "
                f"state, not the peak. Check explained_variance in the "
                f"training log (near-zero suggests instability, not just "
                f"early training) before deciding whether to retrain with "
                f"different hyperparameters (e.g. lower learning_rate, "
                f"larger batch_size) rather than just running longer. The "
                f"checkpoint closest to episode {peak_episode} in "
                f"./checkpoints is likely a better artifact than the final "
                f"saved model."
            )
            return

        if abs(total_improvement) < 1e-6:
            print(
                "[trend check] VERDICT: flat -- rolling mean reward barely "
                "moved over the whole run. Before extending timesteps, "
                "double-check this isn't a pipeline problem (e.g. "
                "gridlocked seeds leaking into the pool, reward scale, "
                "action space) rather than 'needs more steps'. See "
                "progress_so_far.md's gridlock-filter section."
            )
            return

        tail_share = tail_improvement / total_improvement
        print(
            f"[trend check] rolling-mean reward moved {total_improvement:.1f} "
            f"over the full run, {tail_improvement:.1f} of that ({tail_share:.0%}) "
            f"in the final 10% of episodes."
        )
        # Baseline: if improvement were spread perfectly evenly across the
        # whole run, the final 10% of episodes would contribute ~10% of the
        # total by construction -- that's a STILL TRENDING curve (e.g. pure
        # linear climb), not a converged one. So the cutoff for "flattening"
        # needs to sit BELOW that even-spread baseline, not above it -- a
        # threshold like 15% would misclassify a textbook linear climb as
        # converged, which defeats the point of this check. 5% is comfortably
        # below the ~10% even-spread baseline, so only curves whose tail
        # genuinely contributes less than its "fair share" count as flattening.
        FLATTENING_TAIL_SHARE = 0.05
        if tail_share > FLATTENING_TAIL_SHARE:
            print(
                "[trend check] VERDICT: still trending -- the tail is "
                "contributing a real share of total improvement, so this "
                "run likely hasn't converged yet. Consider resuming from "
                "the latest checkpoint in ./checkpoints with more "
                "timesteps rather than treating this run as final."
            )
        else:
            print(
                "[trend check] VERDICT: flattening -- the tail is "
                "contributing little further improvement, consistent with "
                "convergence (or at least a plateau) at this timestep "
                "budget. Worth a fixed-timer baseline comparison next "
                "(see progress_so_far.md's Next steps) before assuming "
                "this policy is demo-ready."
            )


def make_env(grid_w, grid_h, max_ticks, seed_pool, filter_gridlock=True):
    # spawn_scale is looked up per grid size rather than hardcoded: a value
    # safe for one grid size can gridlock a different one, since num_hubs
    # stays fixed while hub density (hubs per intersection) drops as the
    # grid grows -- see spawn_scale_lookup.py for the sweep this is built
    # from. Assumes a square grid; safe_spawn_scale is defined in terms of
    # a single side length.
    assert grid_w == grid_h, "safe_spawn_scale lookup assumes a square grid"
    spawn_scale = safe_spawn_scale(grid_w)
    # filter_gridlock defaults True (TrafficGymEnv's own safety net) but the
    # __main__ block below already runs filter_gridlocked_seeds once on the
    # full candidate pool before calling this -- passing filter_gridlock=False
    # there avoids re-probing an already-filtered pool a second time.
    return Monitor(
        TrafficGymEnv(
            grid_w=grid_w,
            grid_h=grid_h,
            max_ticks=max_ticks,
            seed_pool=seed_pool,
            spawn_scale=spawn_scale,
            stall_penalty=3.0,
            filter_gridlock=filter_gridlock,
        )
    )


if __name__ == "__main__":
    GRID_W, GRID_H = 3, 3
    MAX_TICKS = 300
    # Stage-1 previously trained on a single fixed city (seed_pool=[42]) to
    # validate the pipeline mechanically -- see progress_so_far.md, which
    # confirmed real convergence there (-3400 -> -1500 reward over 1000
    # episodes) but flagged random-city training as a harder problem not
    # yet attempted at this timestep budget. This run switches to a real
    # multi-seed pool, filtered through gridlock_filter.py first: without
    # the filter, some fraction of sampled cities in the pool would be
    # gridlocked (no learnable signal, unbounded queues regardless of
    # policy -- see gridlock_filter.py), which would corrupt the reward
    # signal in a way that's easy to misread as "PPO isn't converging"
    # when the actual cause is a subset of unsolvable training cities.
    TOTAL_TIMESTEPS = 300_000  # same budget as the single-city run; increase
    # if the reward curve is still trending up at the end -- a harder,
    # more-varied training distribution may need more steps to show the
    # same visible convergence the single-city run showed.
    CANDIDATE_SEED_POOL = list(range(1, 101))  # 100 candidate city seeds

    spawn_scale = safe_spawn_scale(GRID_W)
    print(f"Filtering {len(CANDIDATE_SEED_POOL)} candidate seeds for gridlock "
          f"at grid={GRID_W}x{GRID_H}, spawn_scale={spawn_scale}...")
    healthy_seeds, gridlocked_seeds, _diag = filter_gridlocked_seeds(
        CANDIDATE_SEED_POOL, GRID_W, GRID_H, spawn_scale
    )
    print(f"  kept {len(healthy_seeds)}/{len(CANDIDATE_SEED_POOL)} seeds "
          f"({len(gridlocked_seeds)} rejected as gridlocked)")
    TRAIN_SEED_POOL = healthy_seeds

    env = make_env(GRID_W, GRID_H, MAX_TICKS, TRAIN_SEED_POOL, filter_gridlock=False)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        device="cpu",  # small MLP: GPU adds transfer overhead, not speed
    )

    reward_logger = EpisodeRewardLogger(log_every=20, out_dir=".")
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path="./checkpoints",
        name_prefix="ppo_traffic_stage1",
    )
    callbacks = CallbackList([reward_logger, checkpoint_cb])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    model.save("ppo_traffic_stage1")
    print("Saved ppo_traffic_stage1.zip")