//! Procedural city generation.
//!
//! Produces a grid-topology road network (so per-intersection NS/EW signal
//! phasing stays well-defined) but layers in the structure that makes a
//! generated city read as a *city* rather than uniform random noise:
//!
//! - **Hotspot clusters**: a handful of intersections act as "hub" points
//!   (downtown cores, business districts). Every intersection gets a
//!   `zone_weight` from distance-decayed influence of the nearest hubs, so
//!   density falls off smoothly away from hubs instead of being drawn
//!   independently per road. Multiple hubs can overlap, creating a busier
//!   corridor between them, the way a real city has more than one dense
//!   area connected by heavy arterial roads.
//! - **Road hierarchy**: arterial / collector / local, determined by the
//!   zone_weight of the two intersections a road connects (hub-to-hub,
//!   hub-to-outskirts, outskirts-to-outskirts) rather than an independent
//!   coin flip per road. This is what makes "arterial" correlate with
//!   actual position in the city instead of being spatially random.
//! - **Variable connectivity**: hub intersections preferentially get extra
//!   shortcut roads (denser downtown connectivity); low-zone_weight
//!   intersections have a chance of losing a redundant grid connection
//!   (sparser outskirts), instead of every intersection having identical
//!   degree.
//!
//! What this deliberately still does NOT model: continuous vehicle physics,
//! one-way/lane-count detail below the arterial/collector/local split,
//! non-grid street layouts, highways, or anything at metro-area scale --
//! this generates a signaled road *network* sized like a dense multi-block
//! neighborhood (grid_w * grid_h intersections), not a full city, and the
//! "city" language in the project is a simplification on top of that.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadClass {
    /// Backbone roads connecting high-density (hub-adjacent) intersections.
    /// Highest capacity and intensity.
    Arterial,
    /// Feeds traffic from local streets into arterials -- one endpoint is
    /// hub-adjacent, the other is not. Medium capacity/intensity.
    Collector,
    /// Everything else: outskirts-to-outskirts local streets. Lowest
    /// capacity and intensity.
    Local,
}

/// Which phase group a road belongs to at its destination intersection.
/// Group 0 = "NS corridor", Group 1 = "EW corridor". Grid roads map naturally;
/// diagonal/shortcut roads are assigned by their dominant axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseGroup {
    NorthSouth,
    EastWest,
}

#[derive(Debug, Clone)]
pub struct Road {
    pub id: usize,
    pub from: usize,
    pub to: usize,
    pub length: f32,
    /// Max vehicles allowed to queue at this road's intersection end,
    /// waiting on the light. Distinct from `transit_capacity` (see below) --
    /// this is the "at the light" cap, not the "on the road" cap.
    pub capacity: usize,
    pub class: RoadClass,
    /// 0.0..=1.0 spawn-rate multiplier. Derived from the zone_weight of the
    /// intersections this road connects (see module docs), not drawn
    /// independently -- this is what gives the city spatially-correlated
    /// density (busy corridors) instead of random per-road noise.
    pub base_intensity: f32,
    pub phase_group: PhaseGroup,
    /// Ticks a vehicle spends physically driving this road before it
    /// reaches the intersection-queue. Derived from `length` and `class`
    /// (arterials move faster per unit length, not just wider). This is
    /// what gives the road a real travel time -- previously a vehicle
    /// crossed any road in exactly one tick regardless of length, so
    /// "longer route" only cost more in the router's planning-time cost
    /// function, never in actual simulated travel time.
    pub travel_ticks: u32,
    /// Max vehicles that can be mid-road ("driving", not yet queued at the
    /// light) simultaneously. Derived from `length` and `class`. Distinct
    /// from `capacity` -- a road can be full of moving traffic without a
    /// single vehicle having reached its queue yet, which is what lets
    /// "busy but flowing" be a real, distinguishable state from "jammed at
    /// the light".
    pub transit_capacity: usize,
    /// Vehicles allowed to advance from this road's queue into the next
    /// road per green tick (previously hardcoded to 1 for every road
    /// regardless of class). Wider/higher-class roads clear more vehicles
    /// per green, which is what makes signal *timing* matter more on busy
    /// arterials than on a one-lane local street.
    pub saturation_flow: usize,
    /// True for a grade-separated bypass (e.g. a flyover) that skips the
    /// destination intersection's phase check entirely -- always flows,
    /// subject only to its own capacity/saturation_flow. False for every
    /// ordinary signaled road. Not yet used by city generation (Phase 2);
    /// present now so `sim.rs`'s phase-check logic has a stable field to
    /// read.
    pub grade_separated: bool,
}

#[derive(Debug, Clone)]
pub struct Intersection {
    pub id: usize,
    pub x: i32,
    pub y: i32,
    /// Road ids that terminate here (i.e. vehicles queue on these approaching this intersection).
    pub incoming: Vec<usize>,
    /// Road ids that start here.
    pub outgoing: Vec<usize>,
    /// 0.0..=1.0 combined influence of the nearest hotspot cluster(s).
    /// 1.0 = right on top of a hub, decaying toward 0 with distance.
    /// Drives road classification, intensity, and connectivity density --
    /// see module docs.
    pub zone_weight: f32,
}

#[derive(Debug, Clone)]
pub struct City {
    pub intersections: Vec<Intersection>,
    pub roads: Vec<Road>,
    pub seed: u64,
    /// Grid coordinates of the hotspot cluster centers used to generate
    /// this city's zone_weight field. Exposed mainly for visualization/
    /// debugging (e.g. drawing hub markers on a rendered map).
    pub hub_centers: Vec<(i32, i32)>,
}

#[derive(Debug, Clone)]
pub struct CityGenParams {
    pub grid_w: usize,
    pub grid_h: usize,
    /// Probability weight for adding non-grid "shortcut" roads (route
    /// diversity). Actual per-intersection shortcut probability also scales
    /// with that intersection's zone_weight -- hubs get proportionally more.
    pub extra_road_prob: f64,
    /// Number of hotspot clusters ("mini-downtowns"). 0 falls back to a
    /// single implicit hub at the grid center.
    pub num_hubs: usize,
    /// How fast zone_weight decays with grid distance from the nearest hub.
    /// Larger = influence reaches further (bigger, gentler-sloped hub
    /// regions); smaller = sharp, localized hubs.
    pub hub_falloff: f32,
    /// Probability a hub-to-hub or hub-adjacent road is classified as
    /// Arterial rather than Collector when zone_weight alone doesn't force
    /// the answer; adds some stochastic variety on top of the geography.
    pub arterial_prob: f64,
    pub arterial_intensity: (f32, f32),
    pub collector_intensity: (f32, f32),
    pub local_intensity: (f32, f32),
    /// Probability an eligible low-zone_weight intersection has one
    /// redundant grid edge (a bidirectional pair) pruned, thinning
    /// outskirts connectivity. An intersection is never pruned below
    /// degree 1 in any direction it already has, so the network stays
    /// connected.
    pub prune_prob: f64,
    /// Number of grade-separated "flyover" road pairs (bidirectional) added
    /// on top of the grid + shortcut network. Each flyover connects two
    /// high-zone_weight intersections directly (skipping the signaled
    /// intersections between them) -- Phase 2, see major_refactor.md §3.
    /// 0 disables flyover generation entirely (e.g. for tests/city sizes
    /// that predate this feature and shouldn't change behavior).
    pub num_flyovers: usize,
    /// Only intersections at or above this zone_weight are eligible
    /// flyover endpoints -- flyovers model bypasses between busy districts,
    /// not shortcuts from/to the outskirts.
    pub flyover_min_zone_weight: f32,
    /// Flyover effective speed (length units per tick), same units as the
    /// per-class `speed` used for travel_ticks. Higher than even Arterial
    /// (22.0) -- that's the whole point of a grade-separated bypass: it's
    /// long in raw `length` (point-to-point across the city) but fast
    /// enough per unit length that its *travel_ticks* ends up competitive
    /// with routing through several signaled intersections.
    pub flyover_speed: f32,
}

impl Default for CityGenParams {
    fn default() -> Self {
        Self {
            grid_w: 8,
            grid_h: 8,
            extra_road_prob: 0.08,
            num_hubs: 3,
            hub_falloff: 2.5,
            arterial_prob: 0.6,
            arterial_intensity: (0.7, 1.0),
            collector_intensity: (0.35, 0.65),
            local_intensity: (0.05, 0.3),
            prune_prob: 0.15,
            num_flyovers: 2,
            flyover_min_zone_weight: 0.55,
            flyover_speed: 40.0,
        }
    }
}

/// Distance-decayed influence of a single hub at grid distance `dist`
/// (Chebyshev-ish via Euclidean here, cheap and smooth enough for this
/// purpose). `falloff` controls how quickly influence drops off.
fn hub_influence(dist: f32, falloff: f32) -> f32 {
    (-dist / falloff.max(0.01)).exp()
}

/// Compute each intersection's zone_weight as the max (not sum) influence
/// across all hubs. Max rather than sum keeps weight in a stable 0..=1
/// range regardless of how many hubs happen to be nearby, while still
/// letting a road *between* two hubs read as busy because both its
/// endpoints individually score high.
fn compute_zone_weights(intersections: &mut [Intersection], hubs: &[(i32, i32)], falloff: f32) {
    if hubs.is_empty() {
        for i in intersections.iter_mut() {
            i.zone_weight = 1.0;
        }
        return;
    }
    for i in intersections.iter_mut() {
        let mut best = 0.0f32;
        for &(hx, hy) in hubs {
            let dx = (i.x - hx) as f32;
            let dy = (i.y - hy) as f32;
            let dist = (dx * dx + dy * dy).sqrt();
            let infl = hub_influence(dist, falloff);
            if infl > best {
                best = infl;
            }
        }
        i.zone_weight = best.clamp(0.0, 1.0);
    }
}

fn classify_road(
    from_weight: f32,
    to_weight: f32,
    rng: &mut StdRng,
    params: &CityGenParams,
) -> RoadClass {
    let combined = (from_weight + to_weight) / 2.0;
    let both_hubby = from_weight > 0.6 && to_weight > 0.6;
    let either_hubby = from_weight > 0.6 || to_weight > 0.6;

    if both_hubby {
        // Hub-to-hub or within the same dense core: almost always arterial,
        // the backbone connecting downtown-like areas.
        if rng.gen_bool(params.arterial_prob.max(0.85)) {
            RoadClass::Arterial
        } else {
            RoadClass::Collector
        }
    } else if either_hubby || combined > 0.35 {
        // One end near a hub, or moderately central overall: collector,
        // with a chance of graduating to arterial for busier connections.
        if rng.gen_bool(params.arterial_prob * combined as f64) {
            RoadClass::Arterial
        } else {
            RoadClass::Collector
        }
    } else {
        RoadClass::Local
    }
}

fn add_road(
    roads: &mut Vec<Road>,
    intersections: &mut [Intersection],
    from: usize,
    to: usize,
    rng: &mut StdRng,
    params: &CityGenParams,
) {
    let from_weight = intersections[from].zone_weight;
    let to_weight = intersections[to].zone_weight;
    let class = classify_road(from_weight, to_weight, rng, params);

    let (lo, hi) = match class {
        RoadClass::Arterial => params.arterial_intensity,
        RoadClass::Collector => params.collector_intensity,
        RoadClass::Local => params.local_intensity,
    };
    // Blend the class-driven range with this specific road's endpoint
    // density so two roads in the same class still differ a little (a
    // collector deep in a hub's shadow runs busier than one barely
    // qualifying), rather than every road of a class being interchangeable.
    let combined_weight = (from_weight + to_weight) / 2.0;
    let base = lo + (hi - lo) * combined_weight;
    let jitter_span = (hi - lo) * 0.3;
    let intensity = (base + rng.gen_range(-jitter_span..=jitter_span)).clamp(0.02, 1.0);

    let capacity = match class {
        RoadClass::Arterial => rng.gen_range(18..28),
        RoadClass::Collector => rng.gen_range(10..18),
        RoadClass::Local => rng.gen_range(5..10),
    };

    let (fx, fy) = (intersections[from].x, intersections[from].y);
    let (tx, ty) = (intersections[to].x, intersections[to].y);
    let dx = (tx - fx).abs();
    let dy = (ty - fy).abs();
    let phase_group = if dy >= dx {
        PhaseGroup::NorthSouth
    } else {
        PhaseGroup::EastWest
    };

    let length = rng.gen_range(80.0..220.0f32);

    // Effective travel speed (length units per tick) by class -- arterials
    // move faster per unit length, not just wider. These are relative
    // units (length itself is arbitrary 80..220), tuned so a typical
    // arterial crosses in a handful of ticks and a typical local street
    // takes noticeably longer per unit length, not just less capacity.
    let speed = match class {
        RoadClass::Arterial => 22.0,
        RoadClass::Collector => 15.0,
        RoadClass::Local => 10.0,
    };
    let travel_ticks = ((length / speed).round() as u32).max(1);

    // Physical space on the road for moving-but-not-yet-queued vehicles,
    // scaled by length (longer road, more room to spread out) and a
    // class-based density factor (arterials pack more vehicles per unit
    // length -- more effective lanes -- than a local street).
    let density_factor = match class {
        RoadClass::Arterial => 0.12,
        RoadClass::Collector => 0.08,
        RoadClass::Local => 0.05,
    };
    let transit_capacity = ((length * density_factor).round() as usize).max(2);

    let saturation_flow = match class {
        RoadClass::Arterial => 3,
        RoadClass::Collector => 2,
        RoadClass::Local => 1,
    };

    let id = roads.len();
    roads.push(Road {
        id,
        from,
        to,
        length,
        capacity,
        class,
        base_intensity: intensity,
        phase_group,
        travel_ticks,
        transit_capacity,
        saturation_flow,
        grade_separated: false,
    });
    intersections[from].outgoing.push(id);
    intersections[to].incoming.push(id);
}

/// Add a single grade-separated flyover road (one direction). Bypasses the
/// normal `add_road` classification entirely -- a flyover is always
/// `RoadClass::Arterial` (highest capacity/saturation_flow bracket, matching
/// what a grade-separated bypass actually is) and always `grade_separated:
/// true`, so `sim.rs`'s phase-check skips it unconditionally regardless of
/// the signal state at its destination intersection. `length` is real
/// point-to-point Euclidean grid distance (not grid-hop distance) since a
/// flyover cuts straight across, and `travel_ticks` is derived from that
/// length at `flyover_speed` -- deliberately fast enough per unit length
/// that a long point-to-point flyover can still out-cost a multi-hop
/// signaled route once real travel time is in the router's cost function
/// (Phase 1 prerequisite -- see major_refactor.md §3).
fn add_flyover(
    roads: &mut Vec<Road>,
    intersections: &mut [Intersection],
    from: usize,
    to: usize,
    rng: &mut StdRng,
    params: &CityGenParams,
) {
    let (fx, fy) = (intersections[from].x, intersections[from].y);
    let (tx, ty) = (intersections[to].x, intersections[to].y);
    let dx = (tx - fx) as f32;
    let dy = (ty - fy) as f32;
    // Real grid units, same scale as ordinary road `length` (80.0..220.0
    // per hop) -- a flyover spanning several hops legitimately has a much
    // larger `length` than any single ordinary road, which is expected and
    // is what `travel_ticks` (below) has to overcome via speed, not by
    // understating the distance.
    let per_hop_unit = 130.0; // midpoint of the 80.0..220.0 ordinary-road range
    let hop_dist = (dx * dx + dy * dy).sqrt().max(1.0);
    let length = hop_dist * per_hop_unit;

    let travel_ticks = ((length / params.flyover_speed.max(1.0)).round() as u32).max(1);

    // Capacity/transit_capacity follow the same Arterial-class formulas as
    // add_road's Arterial branch, so a flyover isn't implausibly roomier
    // than an ordinary arterial per unit length -- its advantage is speed
    // (travel_ticks), not an unrealistically larger vehicle count.
    let capacity = rng.gen_range(18..28);
    let density_factor = 0.12;
    let transit_capacity = ((length * density_factor).round() as usize).max(2);
    let saturation_flow = 3;

    let phase_group = if dy.abs() >= dx.abs() {
        PhaseGroup::NorthSouth
    } else {
        PhaseGroup::EastWest
    };

    let id = roads.len();
    roads.push(Road {
        id,
        from,
        to,
        length,
        capacity,
        class: RoadClass::Arterial,
        base_intensity: params.arterial_intensity.0
            + rng.gen_range(0.0..(params.arterial_intensity.1 - params.arterial_intensity.0)),
        phase_group,
        travel_ticks,
        transit_capacity,
        saturation_flow,
        grade_separated: true,
    });
    intersections[from].outgoing.push(id);
    intersections[to].incoming.push(id);
}

/// Add `num_flyovers` bidirectional grade-separated flyover pairs, each
/// connecting two distinct high-zone_weight intersections (eligible pool:
/// `zone_weight >= flyover_min_zone_weight`). This is what actually models
/// a bypass "between high-zone_weight areas" per major_refactor.md §3 --
/// low-density outskirts intersections are never flyover endpoints, since a
/// bypass connecting two quiet corners wouldn't be a realistic flyover
/// candidate in the first place. Falls back to a no-op if fewer than 2
/// eligible intersections exist (e.g. a tiny grid or a very high
/// `flyover_min_zone_weight`) rather than panicking or lowering the bar
/// silently -- a city that can't support a flyover just doesn't get one.
fn add_flyovers(
    roads: &mut Vec<Road>,
    intersections: &mut [Intersection],
    rng: &mut StdRng,
    params: &CityGenParams,
) {
    if params.num_flyovers == 0 {
        return;
    }
    let eligible: Vec<usize> = intersections
        .iter()
        .filter(|i| i.zone_weight >= params.flyover_min_zone_weight)
        .map(|i| i.id)
        .collect();
    if eligible.len() < 2 {
        return; // no realistic flyover candidates at this city's density
    }

    // Track already-connected pairs (either direction) so we don't waste a
    // flyover slot re-linking two intersections already directly adjacent
    // in the grid -- that's not a bypass, it's a redundant local road.
    let mut used_pairs: Vec<(usize, usize)> = Vec::new();
    let max_attempts = params.num_flyovers * 20; // generous; bail rather than loop forever on a sparse eligible pool
    let mut placed = 0usize;
    let mut attempts = 0usize;

    while placed < params.num_flyovers && attempts < max_attempts {
        attempts += 1;
        let a = eligible[rng.gen_range(0..eligible.len())];
        let b = eligible[rng.gen_range(0..eligible.len())];
        if a == b {
            continue;
        }
        let (fx, fy) = (intersections[a].x, intersections[a].y);
        let (tx, ty) = (intersections[b].x, intersections[b].y);
        let grid_dist = ((tx - fx).abs() + (ty - fy).abs()) as usize;
        if grid_dist < 3 {
            continue; // too close to be a meaningful bypass, not just a redundant grid edge
        }
        let key = (a.min(b), a.max(b));
        if used_pairs.contains(&key) {
            continue;
        }
        used_pairs.push(key);
        add_flyover(roads, intersections, a, b, rng, params);
        add_flyover(roads, intersections, b, a, rng, params);
        placed += 1;
    }
}

/// Generate a new city deterministically from `seed`. Same seed -> same city,
/// which is what lets us hold out seeds for train/test generalization splits.
pub fn generate_city(seed: u64, params: &CityGenParams) -> City {
    let mut rng = StdRng::seed_from_u64(seed);
    let w = params.grid_w;
    let h = params.grid_h;

    let mut intersections: Vec<Intersection> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let id = y * w + x;
            intersections.push(Intersection {
                id,
                x: x as i32,
                y: y as i32,
                incoming: vec![],
                outgoing: vec![],
                zone_weight: 0.0,
            });
        }
    }

    // Place hotspot clusters at random grid positions. num_hubs=0 falls
    // back to a single implicit center hub so small/degenerate grids still
    // get some density gradient rather than uniform weight.
    let hub_centers: Vec<(i32, i32)> = if params.num_hubs == 0 {
        vec![((w as i32) / 2, (h as i32) / 2)]
    } else {
        (0..params.num_hubs)
            .map(|_| {
                (
                    rng.gen_range(0..w.max(1)) as i32,
                    rng.gen_range(0..h.max(1)) as i32,
                )
            })
            .collect()
    };
    compute_zone_weights(&mut intersections, &hub_centers, params.hub_falloff);

    let mut roads: Vec<Road> = Vec::new();

    // Grid roads, both directions -- but with variable connectivity:
    // low-zone_weight (outskirts) intersections have a chance of pruning a
    // redundant grid edge, thinning connectivity away from hubs the way
    // real outskirts are less densely connected than downtown. Never
    // prunes if it would leave an intersection with zero edges in a given
    // direction pair, so the network stays traversable.
    for y in 0..h {
        for x in 0..w {
            let id = y * w + x;
            if x + 1 < w {
                let right = y * w + (x + 1);
                let local_weight = (intersections[id].zone_weight
                    + intersections[right].zone_weight)
                    / 2.0;
                let skip = local_weight < 0.3
                    && rng.gen_bool(params.prune_prob)
                    && x + 2 < w; // never prune the boundary-adjacent edge, keeps grid connected
                if !skip {
                    add_road(&mut roads, &mut intersections, id, right, &mut rng, params);
                    add_road(&mut roads, &mut intersections, right, id, &mut rng, params);
                }
            }
            if y + 1 < h {
                let down = (y + 1) * w + x;
                let local_weight = (intersections[id].zone_weight
                    + intersections[down].zone_weight)
                    / 2.0;
                let skip = local_weight < 0.3
                    && rng.gen_bool(params.prune_prob)
                    && y + 2 < h;
                if !skip {
                    add_road(&mut roads, &mut intersections, id, down, &mut rng, params);
                    add_road(&mut roads, &mut intersections, down, id, &mut rng, params);
                }
            }
        }
    }

    // Sparse shortcut roads for route diversity, weighted toward hubs: a
    // hub-adjacent intersection is more likely to sprout an extra
    // connection than an outskirts one, mirroring denser downtown street
    // grids/cut-throughs vs sparser suburban connectivity.
    let n = w * h;
    if n > 1 {
        for a in 0..n {
            // Scale this intersection's shortcut chance by its own density
            // rather than applying one flat global probability everywhere.
            let density_scale = 0.3 + 1.7 * intersections[a].zone_weight; // 0.3x..2.0x
            for b in 0..n {
                if a == b {
                    continue;
                }
                let p = (params.extra_road_prob * density_scale as f64) / (n as f64);
                if rng.gen_bool(p.min(1.0)) {
                    add_road(&mut roads, &mut intersections, a, b, &mut rng, params);
                }
            }
        }
    }

    // Grade-separated flyovers, added last -- they don't interact with the
    // grid-adjacency or shortcut-density logic above, just connect two
    // already-placed high-zone_weight intersections directly. Phase 2, see
    // major_refactor.md §3.
    add_flyovers(&mut roads, &mut intersections, &mut rng, params);

    City {
        intersections,
        roads,
        seed,
        hub_centers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_city() {
        let params = CityGenParams::default();
        let a = generate_city(42, &params);
        let b = generate_city(42, &params);
        assert_eq!(a.roads.len(), b.roads.len());
        assert_eq!(a.intersections.len(), b.intersections.len());
        assert_eq!(a.hub_centers, b.hub_centers);
        for (ra, rb) in a.roads.iter().zip(b.roads.iter()) {
            assert_eq!(ra.from, rb.from);
            assert_eq!(ra.to, rb.to);
            assert!((ra.base_intensity - rb.base_intensity).abs() < 1e-6);
        }
    }

    #[test]
    fn different_seed_likely_different_city() {
        let params = CityGenParams::default();
        let a = generate_city(1, &params);
        let b = generate_city(2, &params);
        // Not a hard guarantee, but with this many random draws it should differ.
        assert_ne!(a.roads.len(), b.roads.len());
    }

    #[test]
    fn grid_roads_are_bidirectional() {
        let params = CityGenParams {
            grid_w: 3,
            grid_h: 3,
            extra_road_prob: 0.0,
            prune_prob: 0.0, // isolate this test to the shortcut-road count claim
            num_flyovers: 0, // ditto -- flyovers are a separate road-count source (Phase 2)
            ..Default::default()
        };
        let city = generate_city(7, &params);
        // 3x3 grid: 12 undirected edges * 2 directions = 24 roads, no shortcuts.
        assert_eq!(city.roads.len(), 24);
        for i in &city.intersections {
            assert!(!i.outgoing.is_empty());
        }
    }

    #[test]
    fn zone_weights_are_bounded_and_peak_near_hubs() {
        let params = CityGenParams {
            grid_w: 6,
            grid_h: 6,
            num_hubs: 1,
            hub_falloff: 2.0,
            ..Default::default()
        };
        let city = generate_city(3, &params);
        for i in &city.intersections {
            assert!(i.zone_weight >= 0.0 && i.zone_weight <= 1.0);
        }
        let (hx, hy) = city.hub_centers[0];
        let hub_id = (hy as usize) * params.grid_w + hx as usize;
        let hub_weight = city.intersections[hub_id].zone_weight;
        // The hub's own intersection should be at or extremely near peak
        // influence (distance 0 => e^0 = 1.0).
        assert!(hub_weight > 0.99);
        // Some intersection far from the hub should score meaningfully
        // lower than the hub itself, proving the falloff actually falls off.
        let far = city
            .intersections
            .iter()
            .map(|i| i.zone_weight)
            .fold(f32::INFINITY, f32::min);
        assert!(far < hub_weight * 0.5);
    }

    #[test]
    fn arterial_roads_cluster_near_hubs() {
        // With hubs concentrated in one place, arterial-classified roads
        // should skew toward high combined zone_weight rather than being
        // spread uniformly at random -- this is the actual behavioral
        // difference vs the old independent-coin-flip classification.
        let params = CityGenParams {
            grid_w: 6,
            grid_h: 6,
            num_hubs: 1,
            hub_falloff: 1.5,
            prune_prob: 0.0,
            extra_road_prob: 0.0,
            ..Default::default()
        };
        let city = generate_city(21, &params);
        let mut arterial_weight_sum = 0.0f32;
        let mut arterial_count = 0;
        let mut local_weight_sum = 0.0f32;
        let mut local_count = 0;
        for r in &city.roads {
            let w = (city.intersections[r.from].zone_weight
                + city.intersections[r.to].zone_weight)
                / 2.0;
            match r.class {
                RoadClass::Arterial => {
                    arterial_weight_sum += w;
                    arterial_count += 1;
                }
                RoadClass::Local => {
                    local_weight_sum += w;
                    local_count += 1;
                }
                RoadClass::Collector => {}
            }
        }
        assert!(arterial_count > 0, "expected at least one arterial road");
        assert!(local_count > 0, "expected at least one local road");
        let arterial_avg = arterial_weight_sum / arterial_count as f32;
        let local_avg = local_weight_sum / local_count as f32;
        assert!(
            arterial_avg > local_avg,
            "arterial avg zone_weight {} should exceed local avg {}",
            arterial_avg,
            local_avg
        );
    }

    #[test]
    fn flyovers_are_grade_separated_and_arterial() {
        let params = CityGenParams {
            grid_w: 8,
            grid_h: 8,
            num_hubs: 3,
            num_flyovers: 4,
            flyover_min_zone_weight: 0.55,
            ..Default::default()
        };
        let city = generate_city(101, &params);
        let flyovers: Vec<_> = city.roads.iter().filter(|r| r.grade_separated).collect();
        assert!(
            !flyovers.is_empty(),
            "expected at least one flyover on an 8x8/3-hub city"
        );
        for r in &flyovers {
            assert_eq!(
                r.class,
                RoadClass::Arterial,
                "flyovers should always classify as Arterial"
            );
            // Every non-flyover road on a signaled grid has length in
            // 80.0..220.0; a flyover spans multiple hops so should exceed
            // that range's max (confirms it's really point-to-point, not
            // accidentally reusing the short grid-hop length formula).
            assert!(
                r.length > 220.0,
                "flyover length {} should exceed a single grid hop's max",
                r.length
            );
        }
        // Flyovers come in bidirectional pairs.
        assert_eq!(
            flyovers.len() % 2,
            0,
            "flyovers should always be added as bidirectional pairs"
        );
    }

    #[test]
    fn flyover_endpoints_are_high_zone_weight() {
        let params = CityGenParams {
            grid_w: 8,
            grid_h: 8,
            num_hubs: 3,
            num_flyovers: 4,
            flyover_min_zone_weight: 0.55,
            ..Default::default()
        };
        let city = generate_city(101, &params);
        for r in city.roads.iter().filter(|r| r.grade_separated) {
            assert!(
                city.intersections[r.from].zone_weight >= params.flyover_min_zone_weight,
                "flyover origin zone_weight {} below threshold {}",
                city.intersections[r.from].zone_weight,
                params.flyover_min_zone_weight
            );
            assert!(
                city.intersections[r.to].zone_weight >= params.flyover_min_zone_weight,
                "flyover destination zone_weight {} below threshold {}",
                city.intersections[r.to].zone_weight,
                params.flyover_min_zone_weight
            );
        }
    }

    #[test]
    fn zero_flyovers_produces_no_grade_separated_roads() {
        let params = CityGenParams {
            grid_w: 6,
            grid_h: 6,
            num_flyovers: 0,
            ..Default::default()
        };
        let city = generate_city(55, &params);
        assert!(city.roads.iter().all(|r| !r.grade_separated));
    }

    #[test]
    fn flyover_generation_is_deterministic_per_seed() {
        let params = CityGenParams {
            grid_w: 8,
            grid_h: 8,
            num_hubs: 3,
            num_flyovers: 3,
            ..Default::default()
        };
        let a = generate_city(202, &params);
        let b = generate_city(202, &params);
        let fa: Vec<_> = a
            .roads
            .iter()
            .filter(|r| r.grade_separated)
            .map(|r| (r.from, r.to))
            .collect();
        let fb: Vec<_> = b
            .roads
            .iter()
            .filter(|r| r.grade_separated)
            .map(|r| (r.from, r.to))
            .collect();
        assert_eq!(fa, fb);
    }

    #[test]
    fn tiny_grid_with_no_eligible_intersections_skips_flyovers_gracefully() {
        // flyover_min_zone_weight above what any intersection can reach
        // should just skip flyover generation, not panic or infinite-loop.
        let params = CityGenParams {
            grid_w: 3,
            grid_h: 3,
            num_hubs: 1,
            num_flyovers: 5,
            flyover_min_zone_weight: 1.5, // unreachable -- zone_weight is clamped to <= 1.0
            ..Default::default()
        };
        let city = generate_city(9, &params);
        assert!(city.roads.iter().all(|r| !r.grade_separated));
    }

    #[test]
    fn num_hubs_zero_falls_back_to_center_hub() {
        let params = CityGenParams {
            grid_w: 5,
            grid_h: 5,
            num_hubs: 0,
            ..Default::default()
        };
        let city = generate_city(9, &params);
        assert_eq!(city.hub_centers.len(), 1);
        assert_eq!(city.hub_centers[0], (2, 2));
    }
}