use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadClass {
    Arterial,
    Side,
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
    pub capacity: usize,
    pub class: RoadClass,
    /// 0.0..=1.0 spawn-rate multiplier. Arterial roads run higher.
    pub base_intensity: f32,
    pub phase_group: PhaseGroup,
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
}

#[derive(Debug, Clone)]
pub struct City {
    pub intersections: Vec<Intersection>,
    pub roads: Vec<Road>,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct CityGenParams {
    pub grid_w: usize,
    pub grid_h: usize,
    /// Probability weight for adding non-grid "shortcut" roads (route diversity).
    pub extra_road_prob: f64,
    /// Probability a given road is classified as arterial (higher capacity + intensity).
    pub arterial_prob: f64,
    pub arterial_intensity: (f32, f32),
    pub side_intensity: (f32, f32),
}

impl Default for CityGenParams {
    fn default() -> Self {
        Self {
            grid_w: 6,
            grid_h: 6,
            extra_road_prob: 0.08,
            arterial_prob: 0.25,
            arterial_intensity: (0.6, 1.0),
            side_intensity: (0.1, 0.4),
        }
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
    let is_arterial = rng.gen_bool(params.arterial_prob);
    let (lo, hi) = if is_arterial {
        params.arterial_intensity
    } else {
        params.side_intensity
    };
    let intensity = rng.gen_range(lo..=hi);
    let class = if is_arterial {
        RoadClass::Arterial
    } else {
        RoadClass::Side
    };
    let capacity = if is_arterial {
        rng.gen_range(15..25)
    } else {
        rng.gen_range(6..12)
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

    let id = roads.len();
    roads.push(Road {
        id,
        from,
        to,
        length: rng.gen_range(80.0..220.0),
        capacity,
        class,
        base_intensity: intensity,
        phase_group,
    });
    intersections[from].outgoing.push(id);
    intersections[to].incoming.push(id);
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
            });
        }
    }

    let mut roads: Vec<Road> = Vec::new();

    // Grid roads, both directions, so every block has bidirectional traffic.
    for y in 0..h {
        for x in 0..w {
            let id = y * w + x;
            if x + 1 < w {
                let right = y * w + (x + 1);
                add_road(&mut roads, &mut intersections, id, right, &mut rng, params);
                add_road(&mut roads, &mut intersections, right, id, &mut rng, params);
            }
            if y + 1 < h {
                let down = (y + 1) * w + x;
                add_road(&mut roads, &mut intersections, id, down, &mut rng, params);
                add_road(&mut roads, &mut intersections, down, id, &mut rng, params);
            }
        }
    }

    // Sparse shortcut roads for route diversity (so "shortest" isn't always
    // "only"; the agent/router has alternatives to trade off against congestion).
    let n = w * h;
    if n > 1 {
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                if rng.gen_bool(params.extra_road_prob / (n as f64)) {
                    add_road(&mut roads, &mut intersections, a, b, &mut rng, params);
                }
            }
        }
    }

    City {
        intersections,
        roads,
        seed,
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
            ..Default::default()
        };
        let city = generate_city(7, &params);
        // 3x3 grid: 12 undirected edges * 2 directions = 24 roads, no shortcuts.
        assert_eq!(city.roads.len(), 24);
        for i in &city.intersections {
            assert!(!i.outgoing.is_empty());
        }
    }
}