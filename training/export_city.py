"""
Export one procedurally-generated city to JSON for visualization.

Usage:
    python export_city.py --seed 7 --grid 6 --out city.json
    python export_city.py --seed 7 --grid-w 8 --grid-h 5 --out city.json

Two output shapes depending on what your compiled traffic_sim module
exposes:

- If built with `roads` / `intersection_positions` getters (added to
  pybindings/src/lib.rs -- run `maturin develop` in pybindings/ after
  pulling that change to get these), exports the full road graph: every
  intersection's real (x, y) and every road as (from, to, class,
  capacity). This is what city_viewer.html expects for a proper map.
- If those getters aren't present yet (older compiled .so), falls back to
  a grid-position reconstruction (assumes a plain grid_w x grid_h layout,
  intersection i at (i % grid_w, i // grid_w)) with zone_weights only, no
  road graph. city_viewer.html degrades to a heatmap-only view in that
  case -- still useful, just without roads/hubs-as-graph-edges.

Run this from the training/ directory with traffic_sim importable
(matches how gym_env.py / tune_params.py are already run).
"""
import argparse
import json

import traffic_sim


def export_city(seed, grid_w, grid_h, num_hubs=2, hub_falloff=2.5,
                 extra_road_prob=0.08, arterial_prob=0.6, prune_prob=0.15):
    sim = traffic_sim.TrafficSim(
        seed=seed, grid_w=grid_w, grid_h=grid_h,
        num_hubs=num_hubs, hub_falloff=hub_falloff,
        extra_road_prob=extra_road_prob, arterial_prob=arterial_prob,
        prune_prob=prune_prob,
    )
    sim.reset(seed)

    n = sim.num_intersections
    zone_weights = sim.zone_weights
    hub_centers = sim.hub_centers

    has_positions = hasattr(sim, "intersection_positions")
    has_roads = hasattr(sim, "roads")

    if has_positions:
        positions = sim.intersection_positions
    else:
        # Fallback: reconstruct assuming a plain grid_w x grid_h layout.
        # This matches city.rs's actual intersection ordering for the
        # base grid, but won't reflect any pruned/extra edges' effect on
        # numbering if the real generator ever reorders -- it doesn't
        # today, but this is a best-effort fallback, not a guarantee.
        positions = [(i % grid_w, i // grid_w) for i in range(n)]

    intersections = [
        {"id": i, "x": positions[i][0], "y": positions[i][1], "zone_weight": zone_weights[i]}
        for i in range(n)
    ]

    roads = []
    if has_roads:
        class_names = {0: "arterial", 1: "collector", 2: "local"}
        for from_id, to_id, class_code, capacity in sim.roads:
            roads.append({
                "from": from_id,
                "to": to_id,
                "class": class_names.get(class_code, "local"),
                "capacity": capacity,
            })

    data = {
        "seed": seed,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "num_intersections": n,
        "hub_centers": [{"x": hx, "y": hy} for hx, hy in hub_centers],
        "intersections": intersections,
        "roads": roads,
        "has_real_positions": has_positions,
        "has_roads": has_roads,
    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--grid", type=int, help="shorthand for square grid_w=grid_h=N")
    parser.add_argument("--grid-w", type=int)
    parser.add_argument("--grid-h", type=int)
    parser.add_argument("--num-hubs", type=int, default=2)
    parser.add_argument("--out", default="city.json")
    args = parser.parse_args()

    if args.grid:
        grid_w = grid_h = args.grid
    elif args.grid_w and args.grid_h:
        grid_w, grid_h = args.grid_w, args.grid_h
    else:
        parser.error("pass either --grid N (square) or both --grid-w and --grid-h")

    data = export_city(args.seed, grid_w, grid_h, num_hubs=args.num_hubs)

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    if data["has_roads"]:
        print(f"Exported {data['num_intersections']} intersections, "
              f"{len(data['roads'])} roads -> {args.out}")
    else:
        print(f"Exported {data['num_intersections']} intersections "
              f"(zone_weights only, no road graph -- rebuild with "
              f"`maturin develop` in pybindings/ to get roads) -> {args.out}")