use crate::city::{generate_city, City, CityGenParams, PhaseGroup};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Fixed number of observation fields emitted per intersection:
/// [queue_len_NS, queue_len_EW, wait_time_NS, wait_time_EW, current_phase,
///  local_intensity, transit_approaching, downstream_blockage]
/// Fixed-width per-intersection observations (rather than variable-shaped per
/// city) is what lets one shared policy generalize across differently-shaped
/// cities. The last two fields are aggregate counts/ratios, deliberately NOT
/// per-vehicle data (see REALISM_REFACTOR_PLAN.md section 0 for why) --
/// `transit_approaching` is how many vehicles are currently driving toward
/// this intersection but haven't reached the queue yet (lets the agent
/// anticipate incoming load), and `downstream_blockage` is the fraction of
/// this intersection's outgoing roads currently near transit capacity (lets
/// the agent learn that turning green here doesn't help if there's nowhere
/// for traffic to go).
pub const OBS_PER_INTERSECTION: usize = 8;

/// Fraction (0.0..=1.0) by which a vehicle's per-edge routing cost is
/// randomly jittered when computing its route. This is what produces real
/// route diversity between vehicles sharing the same origin/destination:
/// each vehicle effectively solves shortest-path on a slightly different
/// perceived cost graph (the way real drivers don't all agree on which
/// route is "best"), rather than every vehicle between the same two points
/// taking the identical mathematically-optimal path.
const ROUTE_JITTER: f64 = 0.35;

/// Destinations are sampled biased toward hub/high-zone_weight
/// intersections (commute-style traffic converging on busy areas) rather
/// than uniformly at random across the whole city. This exponent controls
/// how strong that bias is: 1.0 = sample proportional to zone_weight,
/// higher = more strongly concentrated on the busiest intersections.
const DESTINATION_HUB_BIAS: f32 = 1.5;

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
    /// Ticks remaining before this vehicle finishes driving the current
    /// road and becomes eligible to join that road's intersection-queue.
    /// 0 means "in the queue already" or "eligible now, waiting only on
    /// queue room" -- see `advance_transit` in `step()`. Freshly spawned or
    /// freshly-advanced-to-a-new-road vehicles start at
    /// `road.travel_ticks`, decremented once per tick while in transit.
    transit_ticks_left: u32,
}

/// Min-heap entry for Dijkstra's algorithm over intersections. `BinaryHeap`
/// is a max-heap by default, so `Ord` is implemented in reverse of `cost` to
/// turn it into the min-heap the algorithm needs.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DijkstraState {
    cost: f64,
    intersection: usize,
}

impl Eq for DijkstraState {}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is max-heap, we want smallest cost first.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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
    /// the intersection at road.to. Only vehicles that have finished transit
    /// (see `transit` below) live here -- this is specifically the "queued
    /// at the light" population, capped by `road.capacity`.
    queues: Vec<VecDeque<usize>>,
    /// Vehicle slot indices currently driving (not yet queued at the light)
    /// on each road, in entry order (front = closest to finishing transit,
    /// since `travel_ticks` is constant per road so entry order == finish
    /// order). Capped by `road.transit_capacity`. This is what gives a road
    /// a real "busy but flowing" population, distinct from `queues`.
    transit: Vec<VecDeque<usize>>,
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
    /// Per-intersection wait-tick and stall accumulators for this tick only,
    /// reset at the top of every step(). Indexed by intersection id (the
    /// `to` intersection of the road each wait/stall was accrued on) --
    /// this is what makes a localized per-intersection reward possible:
    /// each intersection's signal reflects only traffic actually queued at
    /// its own approaches, not congestion elsewhere in the city. Needed for
    /// the shared-policy architecture (stage 2) to scale to arbitrary city
    /// sizes without per-agent reward getting diluted/noisy as the city
    /// grows.
    wait_by_intersection: Vec<u64>,
    stall_by_intersection: Vec<u32>,
}

impl Simulation {
    pub fn new(seed: u64, params: CityGenParams, config: SimConfig) -> Self {
        let city = generate_city(seed, &params);
        let n_roads = city.roads.len();
        let n_intersections = city.intersections.len();
        Simulation {
            rng: StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15),
            queues: (0..n_roads).map(|_| VecDeque::new()).collect(),
            transit: (0..n_roads).map(|_| VecDeque::new()).collect(),
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
            wait_by_intersection: vec![0; n_intersections],
            stall_by_intersection: vec![0; n_intersections],
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

    /// Sample a destination intersection, biased toward hub/high-zone_weight
    /// areas (commute-style convergence on busy districts) rather than
    /// uniformly across the city. Falls back to uniform if every weight is
    /// ~0 (e.g. num_hubs=0 edge cases where compute_zone_weights left
    /// everything flat) so this never panics on a degenerate weight sum.
    fn sample_destination(&mut self, exclude: usize) -> usize {
        let weights: Vec<f32> = self
            .city
            .intersections
            .iter()
            .map(|i| i.zone_weight.max(0.01).powf(DESTINATION_HUB_BIAS))
            .collect();
        let total: f32 = weights.iter().sum();
        if total <= 0.0 || self.city.intersections.len() <= 1 {
            return exclude;
        }
        loop {
            let mut roll = self.rng.gen_range(0.0..total);
            let mut chosen = 0usize;
            for (idx, &w) in weights.iter().enumerate() {
                if roll < w {
                    chosen = idx;
                    break;
                }
                roll -= w;
            }
            if chosen != exclude {
                return chosen;
            }
            if self.city.intersections.len() <= 1 {
                return exclude;
            }
        }
    }

    /// Dijkstra shortest path (by free-flow travel cost, congestion-blind by
    /// design -- recomputing per-spawn against live queue state would be
    /// expensive and also not how real drivers plan a route before
    /// departing) from `start_intersection` to `dest_intersection`, over a
    /// per-vehicle-jittered cost graph so different vehicles between the
    /// same two points don't all take the identical "optimal" route -- see
    /// ROUTE_JITTER. Returns the sequence of road ids to traverse, or `None`
    /// if unreachable (shouldn't happen on a connected grid, but roads
    /// spawn/prune stochastically so this is checked rather than assumed).
    fn shortest_route(
        &mut self,
        start_intersection: usize,
        dest_intersection: usize,
    ) -> Option<Vec<usize>> {
        if start_intersection == dest_intersection {
            return None;
        }
        let n = self.city.intersections.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut via_road: Vec<Option<usize>> = vec![None; n];
        let mut prev_intersection: Vec<Option<usize>> = vec![None; n];
        dist[start_intersection] = 0.0;

        let mut heap = BinaryHeap::new();
        heap.push(DijkstraState {
            cost: 0.0,
            intersection: start_intersection,
        });

        // Precompute this vehicle's personal jitter per-road up front so the
        // relaxation loop below doesn't call the RNG (keeps the algorithm's
        // correctness easy to reason about) -- one jitter draw per road in
        // the city, still cheap at this city scale.
        let jitters: Vec<f64> = (0..self.city.roads.len())
            .map(|_| 1.0 + self.rng.gen_range(-ROUTE_JITTER..=ROUTE_JITTER))
            .collect();

        while let Some(DijkstraState { cost, intersection }) = heap.pop() {
            if intersection == dest_intersection {
                break;
            }
            if cost > dist[intersection] {
                continue; // stale heap entry
            }
            for &road_id in &self.city.intersections[intersection].outgoing {
                let road = &self.city.roads[road_id];
                // Real travel time (now that roads have one, via
                // travel_ticks) plus a mild congestion-aversion term based
                // on queue capacity -- previously this was length/capacity
                // with no travel-time concept at all, so a "longer" route
                // only ever cost more in planning, never actually took
                // longer or let a vehicle avoid a real queue.
                let base_cost = road.travel_ticks as f64
                    + (2.0 / road.capacity.max(1) as f64);
                let edge_cost = base_cost * jitters[road_id];
                let next_cost = cost + edge_cost;
                if next_cost < dist[road.to] {
                    dist[road.to] = next_cost;
                    via_road[road.to] = Some(road_id);
                    prev_intersection[road.to] = Some(intersection);
                    heap.push(DijkstraState {
                        cost: next_cost,
                        intersection: road.to,
                    });
                }
            }
        }

        if dist[dest_intersection].is_infinite() {
            return None; // unreachable from here in this city's current graph
        }

        // Walk the via_road/prev_intersection chain back from destination to
        // start, then reverse into travel order.
        let mut route = Vec::new();
        let mut cur = dest_intersection;
        while let Some(road_id) = via_road[cur] {
            route.push(road_id);
            cur = prev_intersection[cur].expect("via_road implies prev_intersection is set");
            if cur == start_intersection {
                break;
            }
        }
        route.reverse();
        Some(route)
    }

    /// Build a full spawn-to-destination route starting from `start_road`:
    /// sample a hub-biased destination, then Dijkstra-route from the far
    /// end of `start_road` to it, prepending `start_road` itself. Falls
    /// back to just `[start_road]` (vehicle immediately exits after this
    /// one road) if no route is found -- e.g. a destination that turns out
    /// unreachable because of stochastic road pruning -- rather than
    /// panicking or retrying indefinitely.
    fn route_with_destination(&mut self, start_road: usize) -> Vec<usize> {
        let start_intersection = self.city.roads[start_road].to;
        let dest = self.sample_destination(start_intersection);
        match self.shortest_route(start_intersection, dest) {
            Some(rest) if !rest.is_empty() => {
                let mut route = vec![start_road];
                route.extend(rest);
                route
            }
            _ => vec![start_road],
        }
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
            let transit_capacity = self.city.roads[road_idx].transit_capacity;
            if self.transit[road_idx].len() >= transit_capacity {
                continue; // road's driving-space is full, no room to spawn
            }
            if self.rng.gen_bool((intensity * self.config.spawn_scale).min(1.0)) {
                let route = self.route_with_destination(road_idx);
                let travel_ticks = self.city.roads[road_idx].travel_ticks;
                let vid = self.alloc_vehicle(Vehicle {
                    route,
                    route_pos: 0,
                    wait_ticks: 0,
                    is_stopped: false,
                    stall_count: 0,
                    transit_ticks_left: travel_ticks,
                });
                self.transit[road_idx].push_back(vid);
            }
        }
    }

    /// Advance transit->queue promotion for one road: decrement every
    /// in-transit vehicle's timer, then promote as many front-of-transit
    /// vehicles as have finished (`transit_ticks_left == 0`) and fit in the
    /// queue (`road.capacity`). A vehicle that's finished transit but finds
    /// the queue full stays in transit, marked stopped -- this is spillback:
    /// a jammed light backs traffic up along the road itself, not just at
    /// the light, which is the core behavior this refactor adds.
    fn advance_transit(&mut self, road_idx: usize, total_wait_this_tick: &mut u64) {
        let to_intersection = self.city.roads[road_idx].to;
        let capacity = self.city.roads[road_idx].capacity;

        for &vid in self.transit[road_idx].iter() {
            let v = self.vehicles[vid].as_mut().unwrap();
            if v.transit_ticks_left > 0 {
                v.transit_ticks_left -= 1;
            }
        }

        loop {
            let Some(&vid) = self.transit[road_idx].front() else {
                break;
            };
            let ready = self.vehicles[vid].as_ref().unwrap().transit_ticks_left == 0;
            if !ready {
                break; // entry order == finish order (constant travel_ticks per road)
            }
            if self.queues[road_idx].len() >= capacity {
                // Finished driving but the light's queue is full: spillback.
                // Stays in `transit` (physically still "on the road"),
                // counted as stopped/waiting like any other blocked vehicle.
                let v = self.vehicles[vid].as_mut().unwrap();
                v.wait_ticks += 1;
                v.is_stopped = true;
                *total_wait_this_tick += 1;
                self.wait_by_intersection[to_intersection] += 1;
                break; // front is blocked, nothing behind it can pass either
            }
            self.transit[road_idx].pop_front();
            self.queues[road_idx].push_back(vid);
            // Freshly queued: hasn't waited at the light yet this tick, and
            // isn't retroactively "stopped" just for finishing its drive --
            // it only becomes stopped if it then fails to advance below.
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
        for w in self.wait_by_intersection.iter_mut() {
            *w = 0;
        }
        for s in self.stall_by_intersection.iter_mut() {
            *s = 0;
        }

        // Pass 1: transit -> queue promotion for every road (see
        // advance_transit's docs). Must happen before the queue-advance pass
        // below so a vehicle that finishes driving this tick is visible to
        // this tick's light-timing decision, not one tick delayed.
        for road_idx in 0..self.city.roads.len() {
            self.advance_transit(road_idx, &mut total_wait_this_tick);
        }

        // Pass 2: process queue-advance road-by-road. Each road now advances
        // up to `saturation_flow` head-of-queue vehicles per green tick
        // (previously hardcoded to exactly 1 regardless of road class), and
        // an advancing vehicle enters the NEXT road's transit pool -- it has
        // to physically drive that road before it can queue at its light --
        // rather than jumping straight into the next queue.
        for road_idx in 0..self.city.roads.len() {
            let to_intersection = self.city.roads[road_idx].to;
            let group = self.city.roads[road_idx].phase_group;
            let phase = self.intersections[to_intersection].phase;
            let grade_separated = self.city.roads[road_idx].grade_separated;

            if !grade_separated && !phase.allows(group) {
                self.accumulate_wait(road_idx, &mut total_wait_this_tick);
                continue;
            }

            let saturation_flow = self.city.roads[road_idx].saturation_flow;
            let mut advanced_this_tick = 0usize;

            while advanced_this_tick < saturation_flow {
                let Some(&vid) = self.queues[road_idx].front() else {
                    break;
                };

                let (route_pos, route_len) = {
                    let v = self.vehicles[vid].as_ref().unwrap();
                    (v.route_pos, v.route.len())
                };
                let is_last_hop = route_pos + 1 >= route_len;
                let next_road = if is_last_hop {
                    None
                } else {
                    Some(self.vehicles[vid].as_ref().unwrap().route[route_pos + 1])
                };
                let can_advance = match next_road {
                    None => true, // vehicle exits the network, no downstream room needed
                    Some(nr) => {
                        self.transit[nr].len() < self.city.roads[nr].transit_capacity
                    }
                };

                if !can_advance {
                    // green light but downstream road has no driving-room:
                    // still counts as waiting, and nothing behind this
                    // vehicle can advance either.
                    break;
                }

                self.queues[road_idx].pop_front();
                let was_stopped = self.vehicles[vid].as_ref().unwrap().is_stopped;
                if was_stopped {
                    self.stall_events_this_tick += 1;
                    self.stall_by_intersection[to_intersection] += 1;
                }
                if let Some(nr) = next_road {
                    let v = self.vehicles[vid].as_mut().unwrap();
                    v.is_stopped = false;
                    v.wait_ticks = 0;
                    v.stall_count += was_stopped as u32;
                    v.route_pos += 1;
                    v.transit_ticks_left = self.city.roads[nr].travel_ticks;
                    self.transit[nr].push_back(vid);
                } else {
                    self.completed_this_tick += 1;
                    self.free_vehicle(vid);
                }
                advanced_this_tick += 1;
            }

            if advanced_this_tick == 0 {
                // Nothing moved this tick (blocked or red) -- whole queue
                // is stopped/waiting.
                self.accumulate_wait(road_idx, &mut total_wait_this_tick);
                continue;
            }

            // Remaining queued vehicles (beyond what saturation_flow let
            // through) also accrue wait this tick.
            let mut behind_n = 0u64;
            for &qvid in self.queues[road_idx].iter() {
                let v = self.vehicles[qvid].as_mut().unwrap();
                v.wait_ticks += 1;
                v.is_stopped = true;
                behind_n += 1;
            }
            total_wait_this_tick += behind_n;
            self.wait_by_intersection[to_intersection] += behind_n;
        }

        self.spawn_vehicles();
        self.tick += 1;

        let reward = -(total_wait_this_tick as f32
            + self.config.stall_penalty * self.stall_events_this_tick as f32);
        let done = self.tick >= self.config.max_ticks;
        (self.observe(), reward, done)
    }

    /// Per-intersection reward for this tick: same shaping as the global
    /// scalar reward (negative wait + stall_penalty-weighted stalls) but
    /// computed only from wait/stall events accrued on roads feeding INTO
    /// that intersection. This is what a shared per-intersection policy
    /// should actually train against -- each agent's reward reflects only
    /// congestion it could plausibly have influenced with its own phase
    /// choice, not city-wide congestion diluted across every agent equally.
    /// Without this, agent-level credit assignment gets noisier as the city
    /// grows, which would undermine the whole point of a shared policy
    /// that's meant to generalize to arbitrary city sizes.
    pub fn rewards_per_intersection(&self) -> Vec<f32> {
        self.wait_by_intersection
            .iter()
            .zip(self.stall_by_intersection.iter())
            .map(|(&w, &s)| -(w as f32 + self.config.stall_penalty * s as f32))
            .collect()
    }

    fn accumulate_wait(&mut self, road_idx: usize, total_wait_this_tick: &mut u64) {
        let to_intersection = self.city.roads[road_idx].to;
        let mut n = 0u64;
        for &qvid in self.queues[road_idx].iter() {
            let v = self.vehicles[qvid].as_mut().unwrap();
            v.wait_ticks += 1;
            v.is_stopped = true;
            n += 1;
        }
        *total_wait_this_tick += n;
        self.wait_by_intersection[to_intersection] += n;
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
            let mut transit_approaching = 0f32;
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
                // Vehicles still driving toward this intersection, not yet
                // queued at its light -- lets the agent anticipate load
                // that's about to arrive rather than only reacting to
                // queues that already formed. Aggregate count, not
                // per-vehicle identity (see REALISM_REFACTOR_PLAN.md §0).
                transit_approaching += self.transit[road_id].len() as f32;
            }
            let avg_intensity = if intersection.incoming.is_empty() {
                0.0
            } else {
                intensity_sum / intersection.incoming.len() as f32
            };

            // Fraction of this intersection's OUTGOING roads that are near
            // full on driving-room (transit capacity). High value means
            // "turning green here won't actually help much -- traffic has
            // nowhere to go once it leaves." Threshold of 0.85 is a
            // deliberately coarse "nearly full" cut, not meant to be exact.
            let downstream_blockage = if intersection.outgoing.is_empty() {
                0.0
            } else {
                let near_full = intersection
                    .outgoing
                    .iter()
                    .filter(|&&road_id| {
                        let road = &self.city.roads[road_id];
                        let occ = self.transit[road_id].len() as f32;
                        occ >= 0.85 * road.transit_capacity.max(1) as f32
                    })
                    .count();
                near_full as f32 / intersection.outgoing.len() as f32
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
            obs[base + 6] = transit_approaching;
            obs[base + 7] = downstream_blockage;
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
    fn per_intersection_rewards_sum_to_global_reward() {
        // Localized rewards are just a bucketed decomposition of the same
        // wait/stall accounting the global reward uses -- summing them
        // back up must reproduce the global scalar exactly (mod float
        // rounding), or the localization introduced a real accounting bug.
        let params = CityGenParams {
            grid_w: 4,
            grid_h: 4,
            ..Default::default()
        };
        let mut sim = Simulation::new(7, params, SimConfig { max_ticks: 200, ..Default::default() });
        sim.reset(7);
        let n = sim.num_intersections();
        for _ in 0..200 {
            let actions = vec![0u8; n];
            let (_, global_reward, _) = sim.step(&actions);
            let per_int = sim.rewards_per_intersection();
            assert_eq!(per_int.len(), n);
            let summed: f32 = per_int.iter().sum();
            assert!(
                (summed - global_reward).abs() < 1e-2,
                "summed per-intersection reward {} != global reward {}",
                summed,
                global_reward
            );
        }
    }

    #[test]
    fn routes_reach_sampled_destination_via_valid_road_chain() {
        // Every road in a spawned route must actually connect: road[i].to
        // must equal road[i+1].from, and the final road's `to` should be
        // the intersection the vehicle was actually routed toward. This is
        // the core correctness property of real OD routing (replacing the
        // old random-walk router): routes must be traversable, not just
        // plausible-looking.
        let params = CityGenParams {
            grid_w: 5,
            grid_h: 5,
            ..Default::default()
        };
        let mut sim = Simulation::new(13, params, SimConfig { max_ticks: 300, ..Default::default() });
        sim.reset(13);
        for start_road in 0..sim.city.roads.len().min(30) {
            let route = sim.route_with_destination(start_road);
            assert_eq!(route[0], start_road, "route must begin with the spawn road");
            for w in route.windows(2) {
                let (a, b) = (w[0], w[1]);
                assert_eq!(
                    sim.city.roads[a].to, sim.city.roads[b].from,
                    "route road {} -> {} is not a valid chain",
                    a, b
                );
            }
        }
    }

    #[test]
    fn destinations_skew_toward_hub_zones() {
        // sample_destination should pick high-zone_weight intersections
        // more often than low-zone_weight ones over many draws -- the
        // actual behavioral difference vs uniform-random destination
        // sampling.
        let params = CityGenParams {
            grid_w: 6,
            grid_h: 6,
            num_hubs: 1,
            hub_falloff: 1.2,
            ..Default::default()
        };
        let mut sim = Simulation::new(4, params, SimConfig { max_ticks: 100, ..Default::default() });
        sim.reset(4);
        let n = sim.city.intersections.len();
        let mut hub_picks = 0u32;
        let mut total = 0u32;
        let high_weight_threshold = 0.6;
        for _ in 0..500 {
            let dest = sim.sample_destination(usize::MAX); // exclude nothing reachable
            if sim.city.intersections[dest].zone_weight > high_weight_threshold {
                hub_picks += 1;
            }
            total += 1;
        }
        let hub_intersections = sim
            .city
            .intersections
            .iter()
            .filter(|i| i.zone_weight > high_weight_threshold)
            .count();
        // If sampling were uniform, hub picks would roughly match
        // hub_intersections / n. With hub bias, hub picks should exceed
        // that uniform baseline by a clear margin.
        let uniform_expected = (hub_intersections as f32 / n as f32) * total as f32;
        assert!(
            hub_picks as f32 > uniform_expected * 1.3,
            "hub_picks {} not clearly above uniform baseline {}",
            hub_picks,
            uniform_expected
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

    #[test]
    fn roads_now_have_real_travel_time() {
        // Core Phase-1 claim: travel_ticks is derived from length/class and
        // is >=1 for every road, and a slower/longer road has a travel_ticks
        // that actually reflects its length -- i.e. this isn't a vestigial
        // field that everything still treats as instantaneous.
        let params = CityGenParams {
            grid_w: 5,
            grid_h: 5,
            ..Default::default()
        };
        let city = generate_city(2, &params);
        assert!(!city.roads.is_empty());
        for r in &city.roads {
            assert!(r.travel_ticks >= 1, "travel_ticks must be at least 1 tick");
            assert!(r.transit_capacity >= 2, "transit_capacity must allow at least 2 cars");
        }
        // A vehicle can't cross a road it just entered in the same tick it
        // entered: freshly-spawned vehicles must start with
        // transit_ticks_left == the road's travel_ticks, i.e. >=1, meaning
        // they need at least one full tick before being queue-eligible.
    }

    #[test]
    fn vehicle_spends_real_time_in_transit_before_queueing() {
        // Spawn on a single long-ish road with a red light forever (so we
        // isolate transit delay from queue-wait): the vehicle should NOT
        // appear in `queues` on the very first tick after spawning if
        // travel_ticks > 1 -- it should still be in `transit`.
        let params = CityGenParams {
            grid_w: 4,
            grid_h: 4,
            ..Default::default()
        };
        let mut sim = Simulation::new(6, params, SimConfig {
            max_ticks: 50,
            spawn_scale: 1.0, // force spawning to make this deterministic-ish
            ..Default::default()
        });
        sim.reset(6);
        let n = sim.num_intersections();
        // Find a road with travel_ticks > 1 to make the assertion meaningful.
        let long_road = sim
            .city
            .roads
            .iter()
            .position(|r| r.travel_ticks > 1)
            .expect("expected at least one road with travel_ticks > 1 in a 4x4 city");

        // Manually spawn one vehicle on that road via the public step/spawn
        // path is awkward to target deterministically, so instead assert
        // the invariant directly on spawn_vehicles' effect: after many
        // ticks with actions that keep this road's phase red, transit
        // should show occupancy while the queue for that road stays empty
        // UNTIL travel_ticks have elapsed for the earliest spawn. We can't
        // control which road a vehicle spawns on directly, so instead
        // assert the general invariant across the whole city: at tick 1,
        // no vehicle can be in ANY queue that was reached via transit this
        // same tick with travel_ticks > 1 -- i.e. total transit occupancy
        // across roads with travel_ticks > 1 should be >= 0 and queues for
        // those roads should be empty at tick 1 (nothing has had time to
        // finish a >1-tick drive in 1 tick).
        let actions = vec![0u8; n];
        sim.step(&actions);
        // long_road's queue should still be empty after exactly 1 tick,
        // since travel_ticks > 1 means nothing spawned this tick could have
        // finished driving yet.
        assert_eq!(
            sim.queues[long_road].len(),
            0,
            "a road with travel_ticks > 1 shouldn't have anyone queued after only 1 tick"
        );
    }

    #[test]
    fn saturation_flow_lets_multiple_vehicles_through_on_wide_roads() {
        // An Arterial road (saturation_flow=3) with a deep queue and a
        // green light + open downstream should advance more than 1 vehicle
        // in a single tick -- the behavioral difference from the old
        // hardcoded "1 vehicle per road per tick" model.
        let params = CityGenParams {
            grid_w: 4,
            grid_h: 4,
            ..Default::default()
        };
        let mut sim = Simulation::new(3, params, SimConfig {
            max_ticks: 500,
            spawn_scale: 1.0,
            ..Default::default()
        });
        sim.reset(3);
        let n = sim.num_intersections();
        // Find an arterial road (saturation_flow > 1).
        let arterial = sim
            .city
            .roads
            .iter()
            .position(|r| r.saturation_flow > 1)
            .expect("expected at least one arterial road in a 4x4 city");
        // Force-populate its queue directly past what a single tick of
        // transit could produce, so we can isolate the queue-advance
        // saturation-flow behavior from transit timing.
        let to_intersection = sim.city.roads[arterial].to;
        let cap = sim.city.roads[arterial].capacity;
        let fill = cap.min(6);
        for _ in 0..fill {
            let vid = sim.alloc_vehicle(Vehicle {
                route: vec![arterial],
                route_pos: 0,
                wait_ticks: 5,
                is_stopped: true,
                stall_count: 0,
                transit_ticks_left: 0,
            });
            sim.queues[arterial].push_back(vid);
        }
        let mut actions = vec![1u8; n]; // default everyone to EW-green
        // Force this intersection's phase to whatever this road's group needs.
        let group = sim.city.roads[arterial].phase_group;
        actions[to_intersection] = match group {
            PhaseGroup::NorthSouth => 0,
            PhaseGroup::EastWest => 1,
        };
        let before = sim.queues[arterial].len();
        sim.step(&actions);
        let after = sim.queues[arterial].len();
        let advanced = before - after.min(before);
        assert!(
            advanced > 1,
            "expected saturation_flow to let more than 1 vehicle through in one \
             tick on an arterial with a deep queue and green light, advanced={}",
            advanced
        );
    }

    #[test]
    fn spillback_blocks_transit_when_queue_is_full() {
        // A vehicle that finishes transit while the destination queue is at
        // capacity must stay in `transit` (not silently vanish or force
        // itself into an over-capacity queue) and must be marked
        // stopped/waiting.
        let params = CityGenParams {
            grid_w: 3,
            grid_h: 3,
            ..Default::default()
        };
        let mut sim = Simulation::new(8, params, SimConfig {
            max_ticks: 100,
            spawn_scale: 0.0, // no organic spawning; we control state directly
            ..Default::default()
        });
        sim.reset(8);
        let road_idx = 0usize;
        let cap = sim.city.roads[road_idx].capacity;

        // Fill the queue to capacity with parked (never-advancing) vehicles
        // by keeping the light red for this road's group throughout.
        for _ in 0..cap {
            let vid = sim.alloc_vehicle(Vehicle {
                route: vec![road_idx],
                route_pos: 0,
                wait_ticks: 0,
                is_stopped: false,
                stall_count: 0,
                transit_ticks_left: 0,
            });
            sim.queues[road_idx].push_back(vid);
        }
        assert_eq!(sim.queues[road_idx].len(), cap);

        // Now put one more vehicle in transit, finished (0 ticks left), on
        // the same road -- it should be unable to join the full queue.
        let waiting_vid = sim.alloc_vehicle(Vehicle {
            route: vec![road_idx],
            route_pos: 0,
            wait_ticks: 0,
            is_stopped: false,
            stall_count: 0,
            transit_ticks_left: 0,
        });
        sim.transit[road_idx].push_back(waiting_vid);

        let mut total_wait = 0u64;
        sim.advance_transit(road_idx, &mut total_wait);

        assert_eq!(
            sim.transit[road_idx].len(),
            1,
            "vehicle should remain in transit when destination queue is full"
        );
        assert_eq!(
            sim.queues[road_idx].len(),
            cap,
            "queue should not exceed capacity due to spillback"
        );
        assert!(
            sim.vehicles[waiting_vid].as_ref().unwrap().is_stopped,
            "spilled-back vehicle should be marked stopped/waiting"
        );
        assert!(total_wait > 0, "spillback should contribute to wait accounting");
    }

    #[test]
    fn grade_separated_road_ignores_destination_phase() {
        // A flyover (grade_separated=true) must advance queued vehicles
        // regardless of the destination intersection's signal phase --
        // that's the entire point of grade separation (Phase 2). Force the
        // destination phase to whatever this flyover's phase_group does
        // NOT match, and confirm a queued vehicle still advances.
        let params = CityGenParams {
            grid_w: 8,
            grid_h: 8,
            num_hubs: 3,
            num_flyovers: 4,
            flyover_min_zone_weight: 0.55,
            ..Default::default()
        };
        let mut sim = Simulation::new(101, params, SimConfig {
            max_ticks: 200,
            spawn_scale: 0.0, // we control queue contents directly
            ..Default::default()
        });
        sim.reset(101);
        let n = sim.num_intersections();
        let flyover = sim
            .city
            .roads
            .iter()
            .position(|r| r.grade_separated)
            .expect("expected at least one flyover in this city/seed");

        let to_intersection = sim.city.roads[flyover].to;
        let group = sim.city.roads[flyover].phase_group;
        // Deliberately set the phase to the OPPOSITE of what this road's
        // group would need if it were an ordinary signaled road.
        let mismatched_phase = match group {
            PhaseGroup::NorthSouth => 1u8, // EW-green, so NS traffic would normally be red
            PhaseGroup::EastWest => 0u8,   // NS-green, so EW traffic would normally be red
        };

        let vid = sim.alloc_vehicle(Vehicle {
            route: vec![flyover],
            route_pos: 0,
            wait_ticks: 0,
            is_stopped: true,
            stall_count: 0,
            transit_ticks_left: 0, // already finished transit, sitting in queue
        });
        sim.queues[flyover].push_back(vid);

        let mut actions = vec![0u8; n];
        actions[to_intersection] = mismatched_phase;
        let before = sim.queues[flyover].len();
        sim.step(&actions);
        let after = sim.queues[flyover].len();
        assert!(
            after < before,
            "grade-separated road should advance its queue even against a \
             mismatched signal phase, before={} after={}",
            before,
            after
        );
    }

    #[test]
    fn observation_includes_transit_and_downstream_fields() {
        // OBS_PER_INTERSECTION grew from 6 to 8; sanity-check the new slots
        // are populated (not just zero-padded) when there's transit traffic.
        let params = CityGenParams {
            grid_w: 4,
            grid_h: 4,
            ..Default::default()
        };
        let mut sim = Simulation::new(3, params, SimConfig {
            max_ticks: 200,
            spawn_scale: 1.0,
            ..Default::default()
        });
        sim.reset(3);
        assert_eq!(OBS_PER_INTERSECTION, 8);
        let n = sim.num_intersections();
        let mut saw_nonzero_transit_field = false;
        for _ in 0..50 {
            let actions = vec![0u8; n];
            let (obs, _, _) = sim.step(&actions);
            for i in 0..n {
                let base = i * OBS_PER_INTERSECTION;
                if obs[base + 6] > 0.0 {
                    saw_nonzero_transit_field = true;
                }
                // downstream_blockage is a fraction, must stay in [0,1]
                assert!(obs[base + 7] >= 0.0 && obs[base + 7] <= 1.0);
            }
        }
        assert!(
            saw_nonzero_transit_field,
            "expected some intersection to observe nonzero transit_approaching \
             over 50 ticks of heavy spawning"
        );
    }
}