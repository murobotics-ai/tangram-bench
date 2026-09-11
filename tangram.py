"""Seven convex prisms and one square goal. All distances are in metres."""

import numpy as np
import shapely
from shapely.geometry import Polygon

SIDE = 0.20 * np.sqrt(2)  # UNL sheet printed at 200%
THICKNESS = 0.005
INSET = 0.0005
KNOB_WIDTH = 0.02
KNOB_HEIGHT = 0.04
# A true dissection: two large, one medium, two small triangles, square, parallelogram.
TILES = [
    [(0, 1), (1, 1), (0.5, 0.5)],
    [(0, 0), (0, 1), (0.5, 0.5)],
    [(1, 0), (1, 0.5), (0.5, 0)],
    [(1, 1), (1, 0.5), (0.75, 0.75)],
    [(0.25, 0.25), (0.5, 0.5), (0.75, 0.25)],
    [(0.5, 0.5), (0.75, 0.75), (1, 0.5), (0.75, 0.25)],
    [(0, 0), (0.5, 0), (0.75, 0.25), (0.25, 0.25)],
]
COLORS = [
    "1 .35 .05 1",
    ".15 .65 .9 1",
    ".9 .15 .2 1",
    "1 .85 .2 1",
    ".55 .35 .2 1",
    ".1 .35 .95 1",
    ".4 .75 .15 1",
]
# Human names, colour first, matching COLORS; used in language annotations.
NAMES = [
    "orange large triangle",
    "sky-blue large triangle",
    "red medium triangle",
    "yellow small triangle",
    "brown small triangle",
    "blue square",
    "green parallelogram",
]
CENTERS = np.array([Polygon(p).centroid.coords[0] for p in TILES])
# Keep the nominal tiling origins: inset the edges, never explode the packing.
NOMINAL_VERTICES = [(np.array(p) - c) * SIDE for p, c in zip(TILES, CENTERS)]
VERTICES = [
    np.array(Polygon(v).buffer(-INSET, join_style="mitre").exterior.coords[:-1])
    for v in NOMINAL_VERTICES
]


def knob_yaw(vertices):
    edges = np.roll(vertices, -1, axis=0) - vertices
    longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    return float(np.arctan2(longest[1], longest[0]))


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def goal_transform(seed, target="square"):
    """Independent goal random stream; larger figures keep clear of the source."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    center = np.array([0.32, 0.20 if target == "square" else 0.27]) + rng.uniform(-0.015, 0.015, 2)
    yaw = rng.uniform(-np.pi, np.pi)
    return center, yaw


def goal(seed, target="square"):
    from shapes import canonical

    outline, _, _ = canonical(target)
    center, yaw = goal_transform(seed, target)
    return outline @ rotation(yaw).T + center


def square_solution(outline):
    """One public, analytic solution to our square; useful as a planning oracle."""
    r = np.column_stack(((outline[1] - outline[0]) / SIDE, (outline[3] - outline[0]) / SIDE))
    xy = (CENTERS * SIDE) @ r.T + outline[0]
    return xy, float(np.arctan2(r[1, 0], r[0, 0]))


# Pad triangles by repeating the first corner, so GEOS can build all polygons
# in one vectorized call. This changes representation, never piece geometry.
CORNERS = np.stack([np.vstack((v, v[0])) if len(v) == 3 else v for v in VERTICES])


def polygons(poses):
    """Projected convex footprints; accepts (..., 7 pieces, 7 pose values)."""
    poses = np.asarray(poses)
    w, x, y, z = np.moveaxis(poses[..., 3:], -1, 0)
    # Only the top-left 2x2 of the body-to-world rotation is needed for local z=0.
    r00, r01 = 1 - 2 * (y * y + z * z), 2 * (x * y - w * z)
    r10, r11 = 2 * (x * y + w * z), 1 - 2 * (x * x + z * z)
    xy = np.empty((*poses.shape[:-1], 4, 2))
    xy[..., 0] = (
        r00[..., None] * CORNERS[..., 0] + r01[..., None] * CORNERS[..., 1] + poses[..., 0, None]
    )
    xy[..., 1] = (
        r10[..., None] * CORNERS[..., 0] + r11[..., None] * CORNERS[..., 1] + poses[..., 1, None]
    )
    return shapely.convex_hull(shapely.polygons(xy))


def score_batch(poses, velocities, outlines):
    """Score all worlds at once with NumPy/GEOS; inset slab footprints and face-up physical gates.

    Input shapes: (worlds, 7, 7), (worlds, 7, 6), (worlds, vertices, 2).
    Output: one array per metric, each of shape (worlds,).
    """
    poses, velocities, outlines = (
        np.asarray(x, dtype=float) for x in (poses, velocities, outlines)
    )
    n = len(poses)
    if (
        poses.shape != (n, 7, 7)
        or velocities.shape != (n, 7, 6)
        or outlines.ndim != 3
        or outlines.shape[0] != n
        or outlines.shape[1] < 3
        or outlines.shape[2] != 2
        or n == 0
        or not all(np.isfinite(x).all() for x in (poses, velocities, outlines))
    ):
        raise ValueError("Expected finite poses (N,7,7), velocities (N,7,6), goals (N,V,2)")
    if not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1, atol=1e-6, rtol=0):
        raise ValueError("Piece quaternions must have unit norm")
    footprints, targets = polygons(poses), shapely.polygons(outlines)
    if not np.all(shapely.is_valid(targets) & (shapely.area(targets) > 0)):
        raise ValueError("Goals must be valid polygons with positive area")
    union = shapely.union_all(footprints, axis=-1)
    intersection = shapely.area(shapely.intersection(union, targets))
    iou = intersection / shapely.area(shapely.union(union, targets))
    piece_area = shapely.area(footprints)
    in_goal = shapely.area(shapely.intersection(footprints, targets[:, None]))
    piece_coverage = np.divide(
        in_goal, piece_area, out=np.zeros_like(in_goal), where=piece_area > 0
    )
    overlap = np.maximum(
        0, shapely.area(footprints).sum(axis=-1) - shapely.area(union)
    ) / shapely.area(targets)
    flat = np.all(
        (1 - 2 * (poses[..., 4] ** 2 + poses[..., 5] ** 2)) >= np.cos(np.deg2rad(5)), axis=-1
    )
    on_table = np.all(abs(poses[..., 2] - THICKNESS / 2) < 0.002, axis=-1)
    still = np.all(np.linalg.norm(velocities[..., :3], axis=-1) < 0.01, axis=-1)
    still &= np.all(np.linalg.norm(velocities[..., 3:], axis=-1) < 0.1, axis=-1)
    return {
        "iou": iou,
        "coverage": intersection / shapely.area(targets),
        "pieces_in_goal": (piece_coverage >= 0.95).sum(axis=-1),
        "overlap_fraction": overlap,
        "on_table": on_table,
        "flat": flat,
        "still": still,
        "success": on_table & flat & still & (overlap < 0.01) & (iou >= 0.95),
    }


def score(poses, velocities, outline):
    """Convenient single-world version, also used by the geometry tests."""
    batch = score_batch(
        np.asarray(poses)[None], np.asarray(velocities)[None], np.asarray(outline)[None]
    )
    return {name: value[0].item() for name, value in batch.items()}
