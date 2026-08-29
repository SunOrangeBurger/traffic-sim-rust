"""
Stage-1 training smoke test: fixed grid size, standard SB3 PPO, no
shared-policy architecture yet. Goal here is just to confirm the pipeline
trains without crashing and reward trends upward (less negative) over time
-- NOT to produce a demo-ready policy. See gym_env.py's module docstring for
why this doesn't generalize across city shapes yet.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from gym_env import TrafficGymEnv


class EpisodeRewardLogger(BaseCallback):
    """Prints mean episode reward every `log_every` episodes, so you can see
    a trend without digging through SB3's default logging."""

    def __init__(self, log_every=5):
        super().__init__()
        self.log_every = log_every
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


def make_env(grid_w, grid_h, max_ticks, seed_pool):
    return Monitor(
        TrafficGymEnv(
            grid_w=grid_w,
            grid_h=grid_h,
            max_ticks=max_ticks,
            seed_pool=seed_pool,
            spawn_scale=0.06,
            stall_penalty=3.0,
        )
    )


if __name__ == "__main__":
    GRID_W, GRID_H = 3, 3
    MAX_TICKS = 300
    TOTAL_TIMESTEPS = 20_000
    # Diagnostic override: overfit to ONE city first. If reward doesn't
    # improve even here, the problem is in the pipeline, not "PPO needs more
    # episodes to generalize across cities." Swap back to a real seed range
    # once this shows a clear learning trend.
    TRAIN_SEED_POOL = [42]

    env = make_env(GRID_W, GRID_H, MAX_TICKS, TRAIN_SEED_POOL)

    model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=512,
    batch_size=64,
    learning_rate=3e-4,
    device="cpu",
    )

    callback = EpisodeRewardLogger(log_every=5)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save("ppo_traffic_stage1")
    print("Saved ppo_traffic_stage1.zip")