"""
Stage-1 Gym wrapper: FIXED grid size baked into the observation/action space
shape. This validates that the Rust sim <-> PyO3 <-> Gymnasium <-> SB3 chain
works mechanically end-to-end. It does NOT achieve cross-grid-size
generalization -- the policy's input/output dimensions are tied to this
specific grid_w x grid_h, so "generalizes" here only means unseen seeds/
layouts at the SAME size, not unseen city shapes.

The README's actual generalization architecture is a shared per-intersection
policy (fixed 6-feature local observation -> 1 action, applied identically
regardless of city size). That's a planned stage-2 refactor of this file,
not implemented yet -- don't mistake this for the final architecture.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traffic_sim

from gridlock_filter import filter_gridlocked_seeds


class TrafficGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_w=4,
        grid_h=4,
        max_ticks=1000,
        seed_pool=None,
        spawn_scale=0.06,
        stall_penalty=3.0,
        filter_gridlock=True,
    ):
        super().__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.max_ticks = max_ticks
        self.spawn_scale = spawn_scale
        self.stall_penalty = stall_penalty
        # Pool of city seeds sampled on each reset. Pass a single-element
        # list (e.g. [42]) to deliberately overfit one city while debugging;
        # use a large disjoint range at train vs. eval time for a real
        # generalization test.
        raw_seed_pool = seed_pool if seed_pool is not None else list(range(10_000))
        # Gridlocked seeds (see gridlock_filter.py) have no learnable signal
        # -- queues grow unbounded regardless of signal policy, since the
        # network's throughput ceiling has been exceeded rather than
        # timing being suboptimal. Left in the pool they corrupt the
        # reward signal during training and risk a visibly-broken demo if
        # drawn live. Filtering needs >=4 candidate seeds (see
        # gridlock_filter.py docstring) so single-seed debug pools
        # (e.g. seed_pool=[42]) skip the filter automatically.
        if filter_gridlock and len(raw_seed_pool) >= 4:
            healthy, gridlocked, _diag = filter_gridlocked_seeds(
                raw_seed_pool, grid_w, grid_h, spawn_scale
            )
            if gridlocked:
                print(
                    f"[gridlock_filter] rejected {len(gridlocked)}/{len(raw_seed_pool)} "
                    f"seeds as gridlocked for grid={grid_w}x{grid_h} "
                    f"spawn_scale={spawn_scale}: {sorted(gridlocked)}"
                )
            self.seed_pool = healthy
        else:
            self.seed_pool = raw_seed_pool
        if not self.seed_pool:
            raise ValueError(
                "gridlock filter rejected every candidate seed -- spawn_scale "
                "is likely too high for this grid size; check spawn_scale_lookup.py"
            )
        self._rng = np.random.default_rng()

        self.sim = traffic_sim.TrafficSim(
            seed=self.seed_pool[0],
            grid_w=grid_w,
            grid_h=grid_h,
            max_ticks=max_ticks,
            spawn_scale=spawn_scale,
            stall_penalty=stall_penalty,
        )
        n = self.sim.num_intersections
        obs_dim = n * self.sim.obs_per_intersection
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2] * n)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        city_seed = int(self._rng.choice(self.seed_pool)) if seed is None else seed
        obs = self.sim.reset(city_seed)
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        actions = [int(a) for a in np.asarray(action).flatten()]
        obs, reward, done = self.sim.step(actions)
        terminated = bool(done)
        truncated = False
        info = {}
        return np.asarray(obs, dtype=np.float32), float(reward), terminated, truncated, info