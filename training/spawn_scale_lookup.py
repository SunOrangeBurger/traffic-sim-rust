"""
Size-scaled spawn_scale lookup.

Traffic density that's healthy at one grid size can gridlock a larger one:
num_hubs stays fixed regardless of grid_w/grid_h, so as the grid grows, hub
density (hubs per intersection) drops and the same nominal spawn_scale
concentrates onto proportionally fewer high-intensity roads, worsening
congestion at the same nominal spawn rate. Multi-seed sweeps (see
training/tune_params.py) found the largest 0%-gridlock-rate spawn_scale per
grid size:

    grid    safe spawn_scale   mean_active there
    3x3     0.10               54.0
    4x4     0.06                56.6
    5x5     0.04                56.1
    6x6     0.04                72.5

Notably this is NOT a smooth/linear function of grid size -- it drops
sharply from 3x3 to 5x5 then plateaus at 5x5/6x6 rather than continuing to
fall, because num_hubs=2 fixed means the hub-density ratio itself plateaus
across those two sizes. A parametric formula (linear or otherwise) fit to
these 4 points would either undershoot 3x3 or overshoot 6x6 depending on
which points anchor it. This module instead interpolates directly between
the measured points and refuses to extrapolate past the tested range,
since nothing here justifies guessing what e.g. 8x8 needs.

Stage-2 (mixed grid sizes per episode) is the actual consumer of this: it
needs a matching spawn_scale for whatever size it samples, not one global
constant that's overly conservative for small grids and possibly unsafe
for large ones.
"""

# (grid_side, safe_spawn_scale) control points from the multi-seed sweep.
# Assumes square grids (grid_w == grid_h), which is what both stage-1 and
# the current stage-2 plan use; non-square grids aren't covered by the
# sweep this table is built from.
_SAFE_SPAWN_SCALE_POINTS = [
    (3, 0.10),
    (4, 0.06),
    (5, 0.04),
    (6, 0.04),
]

MIN_TESTED_SIZE = _SAFE_SPAWN_SCALE_POINTS[0][0]
MAX_TESTED_SIZE = _SAFE_SPAWN_SCALE_POINTS[-1][0]


def safe_spawn_scale(grid_side: int) -> float:
    """Linearly interpolate the sweep-verified safe spawn_scale for a given
    square grid side length. Clamps to the nearest tested endpoint rather
    than extrapolating outside [3, 6], since we have no gridlock data past
    that range and guessing would silently reintroduce the exact problem
    this table exists to avoid.
    """
    if grid_side <= MIN_TESTED_SIZE:
        return _SAFE_SPAWN_SCALE_POINTS[0][1]
    if grid_side >= MAX_TESTED_SIZE:
        return _SAFE_SPAWN_SCALE_POINTS[-1][1]

    for (x0, y0), (x1, y1) in zip(_SAFE_SPAWN_SCALE_POINTS, _SAFE_SPAWN_SCALE_POINTS[1:]):
        if x0 <= grid_side <= x1:
            if x1 == x0:
                return y0
            t = (grid_side - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    # Unreachable given the clamps above, but fail loudly rather than
    # silently returning a wrong default if the control-point table is
    # ever edited into a non-monotonic/gapped shape.
    raise ValueError(f"grid_side {grid_side} not covered by interpolation table")


if __name__ == "__main__":
    # Quick sanity print, not a full test suite: confirms exact points
    # round-trip and a couple of interpolated/clamped values look sane.
    checks = [
        (3, 0.10),
        (4, 0.06),
        (5, 0.04),
        (6, 0.04),
        (2, 0.10),   # below range -> clamps to smallest tested
        (10, 0.04),  # above range -> clamps to largest tested
        (3.5, 0.08), # interpolated midpoint between 3x3 and 4x4 points
    ]
    for side, expected in checks:
        got = safe_spawn_scale(side)
        status = "ok" if abs(got - expected) < 1e-9 else f"MISMATCH (expected {expected})"
        print(f"grid_side={side:<6} spawn_scale={got:.4f}  {status}")