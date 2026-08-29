use pyo3::prelude::*;
use traffic_sim_core::{CityGenParams, SimConfig, Simulation, OBS_PER_INTERSECTION};

/// Gym-style env: `TrafficSim(seed).reset(seed)` -> obs; `.step(actions)` ->
/// (obs, reward, done). All the actual simulation logic lives in
/// `traffic_sim_core` -- this file only translates types across the FFI
/// boundary, on purpose, so the core stays testable with plain `cargo test`.
#[pyclass]
struct TrafficSim {
    inner: Simulation,
}

#[pymethods]
impl TrafficSim {
    #[new]
    #[pyo3(signature = (
        seed,
        grid_w=6,
        grid_h=6,
        max_ticks=2000,
        extra_road_prob=0.08,
        arterial_prob=0.6,
        num_hubs=2,
        hub_falloff=2.5,
        prune_prob=0.15,
        spawn_scale=0.06,
        stall_penalty=3.0
    ))]
    fn new(
        seed: u64,
        grid_w: usize,
        grid_h: usize,
        max_ticks: u64,
        extra_road_prob: f64,
        arterial_prob: f64,
        // Number of hotspot clusters ("mini-downtowns") the city gets. More
        // hubs = more distinct busy areas rather than one center.
        num_hubs: usize,
        // How far a hub's influence reaches before decaying to near-zero.
        // Larger = broader, gentler-sloped hub regions.
        hub_falloff: f32,
        // Chance a low-density outskirts intersection has a redundant grid
        // edge pruned, thinning connectivity away from hubs.
        prune_prob: f64,
        // Dynamics knob: how often vehicles spawn per tick per unit of a
        // road's base_intensity. Higher = more congested city. This is the
        // one to sweep if you want to see heavier/lighter traffic.
        spawn_scale: f64,
        // Reward-shaping knob only -- does NOT change simulation dynamics.
        // Controls how much the reward penalizes a stall/re-acceleration
        // event relative to a plain wait-tick.
        stall_penalty: f32,
    ) -> Self {
        let params = CityGenParams {
            grid_w,
            grid_h,
            extra_road_prob,
            arterial_prob,
            num_hubs,
            hub_falloff,
            prune_prob,
            ..Default::default()
        };
        let config = SimConfig {
            max_ticks,
            spawn_scale,
            stall_penalty,
        };
        TrafficSim {
            inner: Simulation::new(seed, params, config),
        }
    }

    /// Reset with a (possibly new) seed. Returns the flat observation vector.
    /// Use a held-out seed range at eval time vs. training to actually test
    /// generalization, per the README.
    fn reset(&mut self, seed: u64) -> Vec<f32> {
        self.inner.reset(seed)
    }

    /// actions[i] in {0,1} per intersection: 0 = North-South green, 1 =
    /// East-West green. Returns (obs, reward, done).
    fn step(&mut self, actions: Vec<u8>) -> PyResult<(Vec<f32>, f32, bool)> {
        if actions.len() != self.inner.num_intersections() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} actions (one per intersection), got {}",
                self.inner.num_intersections(),
                actions.len()
            )));
        }
        Ok(self.inner.step(&actions))
    }

    #[getter]
    fn num_intersections(&self) -> usize {
        self.inner.num_intersections()
    }

    #[getter]
    fn obs_per_intersection(&self) -> usize {
        OBS_PER_INTERSECTION
    }

    /// (active_vehicles, completed_this_tick, total_wait_ticks, stall_events_this_tick)
    fn metrics(&self) -> (usize, usize, u64, u32) {
        let m = self.inner.metrics();
        (
            m.active_vehicles,
            m.completed_vehicles,
            m.total_wait_ticks,
            m.stall_events,
        )
    }

    /// Cumulative stall count across vehicles currently in the network --
    /// the pollution/fuel-waste proxy called out in the README.
    fn total_stall_count(&self) -> u32 {
        self.inner.total_stall_count()
    }

    /// Per-intersection reward for the most recent step(): one f32 per
    /// intersection, in the same order as observe()'s per-intersection
    /// blocks. Localized to each intersection's own approach roads --
    /// needed for the shared per-intersection policy (stage 2) so agents
    /// aren't scored on congestion elsewhere in the city.
    fn rewards_per_intersection(&self) -> Vec<f32> {
        self.inner.rewards_per_intersection()
    }

    /// Grid (x, y) coordinates of this city's hotspot cluster centers.
    /// Useful for visualization (e.g. marking downtown areas on a rendered
    /// map) or for inspecting how city generation responded to num_hubs.
    #[getter]
    fn hub_centers(&self) -> Vec<(i32, i32)> {
        self.inner.city.hub_centers.clone()
    }

    /// zone_weight (0.0..=1.0 hub-influence density) for every
    /// intersection, in intersection-id order. Useful for visualization
    /// (coloring intersections by density) or for confirming a city's
    /// realism knobs (num_hubs, hub_falloff) produced the expected spread.
    #[getter]
    fn zone_weights(&self) -> Vec<f32> {
        self.inner
            .city
            .intersections
            .iter()
            .map(|i| i.zone_weight)
            .collect()
    }
}

#[pymodule]
fn traffic_sim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TrafficSim>()?;
    Ok(())
}