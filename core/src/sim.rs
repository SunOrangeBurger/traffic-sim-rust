use crate::city::{generate_city, City, CityGenParams, PhaseGroup};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// Fixed number of observation fields emitted per intersection:
/// [queue_len_NS, queue_len_EW, wait_time_NS, wait_time_EW, current_phase, local_intensity]
/// Fixed-width per-intersection observations (rather than variable-shaped per
/// city) is what lets one shared policy generalize across differently-shaped
/// cities.
pub const OBS_PER_INTERSECTION: usize = 6;

const MIN_ROUTE_HOPS: usize = 3;
const MAX_ROUTE_HOPS: usize = 8;

/// Dynamics/reward knobs that are *not* about city shape (that's
/// `CityGenParams`). Split into its own struct so tuning sweeps can vary
/// these independently of the road graph.
#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub max_ticks: u64,
    /// Scales how often vehicles spawn per tick per unit of road
    /// base_intensity. This is a dynamics knob: it changes how congested the
    /// city actually gets, independent of anything about reward.
    pub spawn_scale: f64,
    /// Reward weight for a stall/re-acceleration event vs. a plain wait-tick.
    /// This is a REWARD-SHAPING knob only -- it does not affect simulation
    /// dynamics at all (vehicles move/queue identically regardless of its
    /// value). It only changes how much the RL reward penalizes stalling
    /// relative to plain queueing delay.
    pub stall_penalty: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            max_ticks: 2000,
            spawn_scale: 0.06,
            stall_penalty: 3.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    NorthSouthGreen,
    EastWestGreen,
}

impl Phase {
    fn from_action(a: u8) -> Self {
        if a == 0 {
            Phase::NorthSouthGreen
        } else {
            Phase::EastWestGreen
        }
    }
    fn allows(&self, group: PhaseGroup) -> bool {
        matches!(
            (self, group),
            (Phase::NorthSouthGreen, PhaseGroup::NorthSouth)
                | (Phase::EastWestGreen, PhaseGroup::EastWest)
        )
    }
}

struct Vehicle {
    route: Vec<usize>, // sequence of road ids
    route_pos: usize,  // index into route: which road it's currently queued/traveling on
    wait_ticks: u32,
    /// true once it has been stationary for >=1 tick since its last move;
    /// used to detect the moving->stopped transition exactly once per stop.
    is_stopped: bool,
    stall_count: u32,
}

struct IntersectionState {
    phase: Phase,
}

/// Per-tick metrics for logging / the demo's training-curve and before/after
/// comparisons.
#[derive(Debug, Clone, Copy, Default)]
pub struct TickMetrics {
    pub active_vehicles: usize,
    pub completed_vehicles: usize,
    pub total_wait_ticks: u64,
    pub stall_events: u32,
}

pub struct Simulation {
    pub city: City,
    params: CityGenParams,
    config: SimConfig,
    rng: StdRng,
    /// queue of vehicle slot indices waiting on each road; front = closest to
    /// the intersection at road.to
    queues: Vec<VecDeque<usize>>,
    /// Slab arena: `None` = free slot, reused by future spawns. This is what
    /// lets us free completed vehicles without invalidating the indices
    /// stored in `queues` (a plain Vec::retain would shift indices and
    /// silently corrupt every queue referencing a later vehicle).
    vehicles: Vec<Option<Vehicle>>,
    free_slots: Vec<usize>,
    intersections: Vec<IntersectionState>,
    tick: u64,
    completed_this_tick: usize,
    stall_events_this_tick: u32,
}

impl Simulation {
    pub fn new(seed: u64, params: CityGenParams, config: SimConfig) -> Self {
        let city = generate_city(seed, &params);
        let n_roads = city.roads.len();
        let n_intersections = city.intersections.len();
        Simulation {
            rng: StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15),
            queues: (0..n_roads).map(|_| VecDeque::new()).collect(),
            vehicles: Vec::new(),
            free_slots: Vec::new(),
            intersections: (0..n_intersections)
                .map(|_| IntersectionState {
                    phase: Phase::NorthSouthGreen,
                })
                .collect(),
            tick: 0,
            completed_this_tick: 0,
            stall_events_this_tick: 0,
            city,
            params,
            config,
        }
    }

    pub fn num_intersections(&self) -> usize {
        self.city.intersections.len()
    }

    /// Reset with a new seed (new city + fresh state). Mirrors Gym's reset().
    pub fn reset(&mut self, seed: u64) -> Vec<f32> {
        *self = Simulation::new(seed, self.params.clone(), self.config);
        self.observe()
    }

    fn random_route(&mut self, start_road: usize) -> Vec<usize> {
        let hops = self.rng.gen_range(MIN_ROUTE_HOPS..=MAX_ROUTE_HOPS);
        let mut route = vec![start_road];
        let mut current_intersection = self.city.roads[start_road].to;
        for _ in 1..hops {
            let outgoing = &self.city.intersections[current_intersection].outgoing;
            if outgoing.is_empty() {
                break;
            }
            let prev_road = *route.last().unwrap();
            // avoid an immediate U-turn back the way we came, when an alternative exists
            let choice: Vec<usize> = outgoing
                .iter()
                .filter(|&&r| self.city.roads[r].to != self.city.roads[prev_road].from)
                .copied()
                .collect();
            let pool = if choice.is_empty() { outgoing.clone() } else { choice };
            let next = pool[self.rng.gen_range(0..pool.len())];
            current_intersection = self.city.roads[next].to;
            route.push(next);
        }
        route
    }

    fn alloc_vehicle(&mut self, v: Vehicle) -> usize {
        if let Some(idx) = self.free_slots.pop() {
            self.vehicles[idx] = Some(v);
            idx
        } else {
            self.vehicles.push(Some(v));
            self.vehicles.len() - 1
        }
    }

    fn free_vehicle(&mut self, idx: usize) {
        self.vehicles[idx] = None;
        self.free_slots.push(idx);
    }

    fn spawn_vehicles(&mut self) {
        for road_idx in 0..self.city.roads.len() {
            let intensity = self.city.roads[road_idx].base_intensity as f64;
            let capacity = self.city.roads[road_idx].capacity;
            if self.queues[road_idx].len() >= capacity {
                continue; // road is full, no room to spawn
            }
            if self.rng.gen_bool((intensity * self.config.spawn_scale).min(1.0)) {
                let route = self.random_route(road_idx);
                let vid = self.alloc_vehicle(Vehicle {
                    route,
                    route_pos: 0,
                    wait_ticks: 0,
                    is_stopped: false,
                    stall_count: 0,
                });
                self.queues[road_idx].push_back(vid);
            }
        }
    }

    /// Advance the simulation by one tick given a per-intersection phase action.
    /// `actions[i]` in {0,1}: 0 = North-South green, 1 = East-West green, for
    /// `self.city.intersections[i]`.
    pub fn step(&mut self, actions: &[u8]) -> (Vec<f32>, f32, bool) {
        assert_eq!(
            actions.len(),
            self.intersections.len(),
            "one action required per intersection"
        );
        for (i, &a) in actions.iter().enumerate() {
            self.intersections[i].phase = Phase::from_action(a);
        }

        self.completed_this_tick = 0;
        self.stall_events_this_tick = 0;
        let mut total_wait_this_tick: u64 = 0;

        // Process movement road-by-road. Each road advances at most its front
        // (head-of-queue) vehicle per tick while its phase group is green --
        // crude on purpose, matching the README's guidance to avoid a full
        // physics model and just track stopped-vs-moving.
        for road_idx in 0..self.city.roads.len() {
            let to_intersection = self.city.roads[road_idx].to;
            let group = self.city.roads[road_idx].phase_group;
            let phase = self.intersections[to_intersection].phase;

            if !phase.allows(group) {
                self.accumulate_wait(road_idx, &mut total_wait_this_tick);
                continue;
            }

            let Some(&vid) = self.queues[road_idx].front() else {
                continue;
            };

            let (route_pos, route_len) = {
                let v = self.vehicles[vid].as_ref().unwrap();
                (v.route_pos, v.route.len())
            };
            let is_last_hop = route_pos + 1 >= route_len;
            let can_advance = if is_last_hop {
                true // vehicle exits the network, no downstream capacity needed
            } else {
                let next_road = self.vehicles[vid].as_ref().unwrap().route[route_pos + 1];
                self.queues[next_road].len() < self.city.roads[next_road].capacity
            };

            if can_advance {
                self.queues[road_idx].pop_front();
                let was_stopped = self.vehicles[vid].as_ref().unwrap().is_stopped;
                if was_stopped {
                    self.stall_events_this_tick += 1;
                }
                if is_last_hop {
                    self.completed_this_tick += 1;
                    self.free_vehicle(vid);
                } else {
                    let v = self.vehicles[vid].as_mut().unwrap();
                    v.is_stopped = false;
                    v.wait_ticks = 0;
                    v.stall_count += was_stopped as u32;
                    v.route_pos += 1;
                    let next_road = v.route[v.route_pos];
                    self.queues[next_road].push_back(vid);
                }
            } else {
                // green light but downstream is jammed: still counts as waiting
                self.accumulate_wait(road_idx, &mut total_wait_this_tick);
                continue;
            }

            // vehicles queued behind the one that just moved also accrue wait
            for &qvid in self.queues[road_idx].iter() {
                let v = self.vehicles[qvid].as_mut().unwrap();
                v.wait_ticks += 1;
                v.is_stopped = true;
                total_wait_this_tick += 1;
            }
        }

        self.spawn_vehicles();
        self.tick += 1;

        let reward = -(total_wait_this_tick as f32
            + self.config.stall_penalty * self.stall_events_this_tick as f32);
        let done = self.tick >= self.config.max_ticks;
        (self.observe(), reward, done)
    }

    fn accumulate_wait(&mut self, road_idx: usize, total_wait_this_tick: &mut u64) {
        for &qvid in self.queues[road_idx].iter() {
            let v = self.vehicles[qvid].as_mut().unwrap();
            v.wait_ticks += 1;
            v.is_stopped = true;
            *total_wait_this_tick += 1;
        }
    }

    pub fn metrics(&self) -> TickMetrics {
        let mut active = 0usize;
        let mut total_wait = 0u64;
        for v in self.vehicles.iter().flatten() {
            active += 1;
            total_wait += v.wait_ticks as u64;
        }
        TickMetrics {
            active_vehicles: active,
            completed_vehicles: self.completed_this_tick,
            total_wait_ticks: total_wait,
            stall_events: self.stall_events_this_tick,
        }
    }

    /// Total stalls accumulated by vehicles currently in the network (a
    /// pollution/fuel-waste proxy per the README).
    pub fn total_stall_count(&self) -> u32 {
        self.vehicles
            .iter()
            .flatten()
            .map(|v| v.stall_count)
            .sum()
    }

    pub fn observe(&self) -> Vec<f32> {
        let mut obs = vec![0.0f32; self.city.intersections.len() * OBS_PER_INTERSECTION];
        for (i, intersection) in self.city.intersections.iter().enumerate() {
            let base = i * OBS_PER_INTERSECTION;
            let mut ns_queue = 0f32;
            let mut ew_queue = 0f32;
            let mut ns_wait = 0f32;
            let mut ew_wait = 0f32;
            let mut intensity_sum = 0f32;
            for &road_id in &intersection.incoming {
                let road = &self.city.roads[road_id];
                let qlen = self.queues[road_id].len() as f32;
                let wait: f32 = self.queues[road_id]
                    .iter()
                    .map(|&vid| self.vehicles[vid].as_ref().unwrap().wait_ticks as f32)
                    .sum();
                match road.phase_group {
                    PhaseGroup::NorthSouth => {
                        ns_queue += qlen;
                        ns_wait += wait;
                    }
                    PhaseGroup::EastWest => {
                        ew_queue += qlen;
                        ew_wait += wait;
                    }
                }
                intensity_sum += road.base_intensity;
            }
            let avg_intensity = if intersection.incoming.is_empty() {
                0.0
            } else {
                intensity_sum / intersection.incoming.len() as f32
            };
            obs[base] = ns_queue;
            obs[base + 1] = ew_queue;
            obs[base + 2] = ns_wait;
            obs[base + 3] = ew_wait;
            obs[base + 4] = match self.intersections[i].phase {
                Phase::NorthSouthGreen => 0.0,
                Phase::EastWestGreen => 1.0,
            };
            obs[base + 5] = avg_intensity;
        }
        obs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_gives_correctly_shaped_observation() {
        let params = CityGenParams {
            grid_w: 3,
            grid_h: 3,
            ..Default::default()
        };
        let mut sim = Simulation::new(1, params, SimConfig { max_ticks: 500, ..Default::default() });
        let obs = sim.reset(1);
        assert_eq!(obs.len(), sim.num_intersections() * OBS_PER_INTERSECTION);
    }

    #[test]
    fn step_runs_and_terminates_at_max_ticks() {
        let params = CityGenParams {
            grid_w: 3,
            grid_h: 3,
            ..Default::default()
        };
        let mut sim = Simulation::new(1, params, SimConfig { max_ticks: 20, ..Default::default() });
        sim.reset(1);
        let n = sim.num_intersections();
        let mut done = false;
        let mut steps = 0;
        while !done {
            let actions = vec![0u8; n];
            let (obs, reward, d) = sim.step(&actions);
            assert_eq!(obs.len(), n * OBS_PER_INTERSECTION);
            assert!(reward <= 0.0);
            done = d;
            steps += 1;
            assert!(steps <= 25, "should terminate at max_ticks");
        }
        assert_eq!(steps, 20);
    }

    #[test]
    fn vehicles_spawn_over_time() {
        let params = CityGenParams {
            grid_w: 4,
            grid_h: 4,
            ..Default::default()
        };
        let mut sim = Simulation::new(9, params, SimConfig { max_ticks: 200, ..Default::default() });
        sim.reset(9);
        let n = sim.num_intersections();
        let mut saw_activity = false;
        for _ in 0..200 {
            let actions = vec![0u8; n];
            sim.step(&actions);
            if sim.metrics().active_vehicles > 0 {
                saw_activity = true;
            }
        }
        assert!(saw_activity, "expected some vehicles to spawn over 200 ticks");
    }

    #[test]
    fn completed_vehicle_slots_are_reused_not_leaked() {
        let params = CityGenParams {
            grid_w: 3,
            grid_h: 3,
            ..Default::default()
        };
        let mut sim = Simulation::new(5, params, SimConfig { max_ticks: 3000, ..Default::default() });
        sim.reset(5);
        let n = sim.num_intersections();
        for _ in 0..3000 {
            let actions = vec![0u8; n];
            sim.step(&actions);
        }
        assert!(
            sim.vehicles.len() < 3000,
            "expected slot reuse to bound arena growth, got {}",
            sim.vehicles.len()
        );
    }

    #[test]
    fn queue_indices_stay_valid_after_completions() {
        // Regression test for the slab-arena fix: after many vehicles
        // complete and free their slots, remaining queued vehicles must
        // still resolve to the correct (non-freed) Vehicle rather than
        // panicking on a stale index.
        let params = CityGenParams {
            grid_w: 3,
            grid_h: 3,
            ..Default::default()
        };
        let mut sim = Simulation::new(11, params, SimConfig { max_ticks: 1000, ..Default::default() });
        sim.reset(11);
        let n = sim.num_intersections();
        for _ in 0..1000 {
            let actions = vec![0u8; n];
            sim.step(&actions);
            sim.observe();
        }
    }
}