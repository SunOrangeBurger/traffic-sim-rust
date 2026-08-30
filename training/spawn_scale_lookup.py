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
    8x8     0.04               219.9

Notably this is NOT a smooth/linear function of grid size -- it drops
sharply from 3x3 to 5x5 then plateaus at 5x5/6x6 rather than continuing to
fall, because num_hubs=2 fixed means the hub-density ratio itself plateaus
across those two sizes. A parametric formula (linear or otherwise) fit to
these 4 points would either undershoot 3x3 or overshoot 6x6 depending on
which points anchor it. This module instead interpolates directly between
the measured points and refuses to extrapolate past the tested range,
since nothing here justifies guessing what e.g. 8x8 needs.

8x8 point added post-Phase-2 (major_refactor.md §3 bumped
CityGenParams::default() to grid_w=grid_h=8, num_hubs=3 -- note num_hubs
changed too, not just grid size, so this point isn't a pure like-for-like
extension of the num_hubs=2 series above it). Validated via
gridlock_filter.py's median-ratio method (the module trusted over
tune_params.py's own growth_ratio verdict -- see errors&findings.md §2 for
why growth_ratio has confirmed false negatives) at 8 seeds, both 0.03 and
0.04 candidates: 0.04 came back clean (no gridlocked seeds, ratios
0.65-1.22, same healthy band as the other rows) and was chosen over 0.03
for matching 6x6's already-validated value rather than introducing a new
number, and for the higher resulting traffic density (more for an RL agent
to actually learn from) with no gridlock cost. **Caveat carried over
directly from the same section of progress_so_far.md that flagged this
same risk for the OLDER points above:** validated at 8 seeds here, same as
this table's other entries -- but progress_so_far.md's own later 50-seed
re-check of those other "safe" values found nonzero residual gridlock
(0-12%) that the original 8-seed sweep missed. This 8x8 point carries that
identical risk and hasn't been re-checked at 50 seeds. Nothing here past
3x3-6x6, 8x8 is genuinely tested; do not interpolate/extrapolate to 7x7 or
anything above 8x8 without running the sweep at that size directly.

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
    (8, 0.04),
]

MIN_TESTED_SIZE = _SAFE_SPAWN_SCALE_POINTS[0][0]
MAX_TESTED_SIZE = _SAFE_SPAWN_SCALE_POINTS[-1][0]

# NOTE: grid_side=7 falls between two tested points (6 and 8) that both
# happen to be 0.04, so safe_spawn_scale(7) returns 0.04 -- but 7x7 itself
# has NOT been swept. That both neighbors agree is a coincidence of this
# particular plateau, not evidence 7x7 is safe. Re-run tune_params.py +
# gridlock_filter.py at 7x7 directly before trusting this for anything
# beyond a rough guess.


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
        (8, 0.04),
        (2, 0.10),   # below range -> clamps to smallest tested
        (10, 0.04),  # above range -> clamps to largest tested
        (3.5, 0.08), # interpolated midpoint between 3x3 and 4x4 points
        (7, 0.04),   # interpolated between 6x6 and 8x8 (both 0.04) -- see NOTE above, not itself tested
    ]
    for side, expected in checks:
        got = safe_spawn_scale(side)
        status = "ok" if abs(got - expected) < 1e-9 else f"MISMATCH (expected {expected})"
        print(f"grid_side={side:<6} spawn_scale={got:.4f}  {status}")