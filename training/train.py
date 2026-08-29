"""
Stage-1 training: fixed grid size, standard SB3 PPO, single fixed city
(no shared-policy architecture yet -- see gym_env.py's module docstring).

This version is the LONGER convergence run: after the 20k-step smoke test
showed a real (if noisy) improving trend, this bumps timesteps way up,
forces CPU (SB3 warns MlpPolicy-on-GPU is usually slower, not faster, for
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
        plt.title("PPO training curve (single fixed city)")
        plt.legend()
        plt.tight_layout()
        png_path = os.path.join(self.out_dir, "reward_curve.png")
        plt.savefig(png_path, dpi=120)
        print(f"Saved {png_path}")


def make_env(grid_w, grid_h, max_ticks, seed_pool):
    # spawn_scale is looked up per grid size rather than hardcoded: a value
    # safe for one grid size can gridlock a different one, since num_hubs
    # stays fixed while hub density (hubs per intersection) drops as the
    # grid grows -- see spawn_scale_lookup.py for the sweep this is built
    # from. Assumes a square grid; safe_spawn_scale is defined in terms of
    # a single side length.
    assert grid_w == grid_h, "safe_spawn_scale lookup assumes a square grid"
    spawn_scale = safe_spawn_scale(grid_w)
    return Monitor(
        TrafficGymEnv(
            grid_w=grid_w,
            grid_h=grid_h,
            max_ticks=max_ticks,
            seed_pool=seed_pool,
            spawn_scale=spawn_scale,
            stall_penalty=3.0,
        )
    )


if __name__ == "__main__":
    GRID_W, GRID_H = 3, 3
    MAX_TICKS = 300
    # Real convergence run, not a smoke test -- 15x the previous budget.
    # At ~450-600 fps on CPU for this tiny env/network, expect roughly
    # 8-12 minutes. Bump higher if the curve is still trending at the end.
    TOTAL_TIMESTEPS = 300_000
    TRAIN_SEED_POOL = [42]  # still deliberately overfitting ONE city

    env = make_env(GRID_W, GRID_H, MAX_TICKS, TRAIN_SEED_POOL)

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