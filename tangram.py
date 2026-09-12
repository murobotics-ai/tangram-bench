"""Seven convex prisms and one square goal. All distances are in metres."""

import numpy as np
import shapely
from shapely.geometry import Polygon

SIDE = 0.20 * np.sqrt(2)  # UNL sheet printed at 200%
THICKNESS = 0.005
INSET = 0.0005
KNOB_WIDTH = 0.02
KNOB_HEIGHT = 0.04
# Success tolerance: 5 mm per piece. The benchmark asks whether a policy infers
# the figure and the order of the pieces, not whether it places slabs to a
# millimetre. Calibrated on certificate layouts of all four figures with every
# piece displaced by 5 mm in a random direction and turned up to 3 degrees (1,200
# samples): IoU median 0.915, minimum 0.875; footprint overlap median 2.3%,
# maximum 5.6%; 1 failure in 10,000 samples. At 10 mm the IoU median is 0.85.
# A rigid shift of the whole assembly by 10 mm still passes on most figures: the
# test is about the assembly, not its exact spot. A slab resting on a
# neighbour's edge sits 2-3 mm high and tilts about 2 degrees.
IOU_THRESHOLD = 0.87
OVERLAP_THRESHOLD = 0.06
# The IoU is dominated by the large pieces; this gate keeps every piece, small ones
# included, within the tolerance: at 5 mm the least-covered piece is 0.875 inside.
PIECE_COVERAGE = 0.85
TABLE_TOLERANCE = 0.004  # Body origin within this of half-thickness above the table.
FLAT_DEGREES = 8  # Local z axis within this of upward vertical.
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


# Scene design. Nothing about a scene is drawn at random: the seed indexes a
# documented grid of rotations and offsets, so a dataset's coverage can be
# stated exactly and the dev split can ask for rotations never seen in training.
GOAL_YAWS = 12  # goal silhouette yaw on a 30 degree grid
SOURCE_YAWS = 8  # packed square yaw on a 45 degree grid
GOAL_OFFSETS = [(x, y) for x in (-0.015, 0.0, 0.015) for y in (-0.015, 0.0, 0.015)]
SOURCE_OFFSETS = [(x, y) for x in (-0.03, 0.0, 0.03) for y in (-0.03, 0.0, 0.03)]
SOURCE_CENTER = np.array([0.36, -0.30])
SPLIT_SIZE = 100000  # seed // SPLIT_SIZE: 0 train, 1 dev, 2 test (benchmark.SPLITS)
# Workspace: every piece centre, at the source and at the goal, lies between these
# radii from the robot base. Closer than 0.28 m the elbow folds against its stop and
# the forearm meets the shoulder column; the centres below keep every designed
# scene inside (checked by a test over the train and dev grids).
WORKSPACE = (0.28, 0.66)


def goal_center(target):
    return np.array([0.35, 0.28 if target == "square" else 0.31])


def piece_radii(scene, target):
    """Distances from the base of every piece centre at the goal and at the source."""
    from shapes import solution

    goal_xy = np.array(solution(target, 0, scene))[:, :2]
    source = ((CENTERS - 0.5) * SIDE) @ rotation(scene["source_yaw"]).T + scene["source_center"]
    return np.linalg.norm(np.vstack([goal_xy, source]), axis=1)


def layout(seed, target="square"):
    """The scene for a seed: goal and source pose, plus the design indices behind them.

    Within a split, seed n takes goal yaw n mod 12 on the 30 degree grid; the
    goal offset, source yaw and source offset advance with strides coprime to
    their grid sizes, so 60 seeds visit every value of each grid. The four
    indices share the seed counter, so the joint sequence repeats every
    lcm(12, 9, 8, 9) = 72 seeds: a split holds 72 distinct scenes per figure,
    and seed n + 72 is the same scene as seed n. The dev split rotates the goal
    a further 15 degrees, half a grid step: poses between the training
    rotations, never equal to one.
    """
    split, n = divmod(int(seed), SPLIT_SIZE)
    goal_yaw = 2 * np.pi / GOAL_YAWS * (n % GOAL_YAWS) + (np.pi / GOAL_YAWS if split == 1 else 0)
    goal_offset = GOAL_OFFSETS[(n * 5) % len(GOAL_OFFSETS)]
    source_yaw = 2 * np.pi / SOURCE_YAWS * ((n * 3) % SOURCE_YAWS)
    source_offset = SOURCE_OFFSETS[(n * 7 + 2) % len(SOURCE_OFFSETS)]
    return {
        "goal_center": goal_center(target) + goal_offset,
        "goal_yaw": float(goal_yaw),
        "source_center": SOURCE_CENTER + source_offset,
        "source_yaw": float(source_yaw),
    }


def describe_layout(scene):
    """Degrees and millimetres, for index rows and result files."""
    return {
        "goal_yaw_deg": round(float(np.degrees(scene["goal_yaw"])) % 360, 1),
        "goal_center_mm": [round(float(v) * 1000, 1) for v in scene["goal_center"]],
        "source_yaw_deg": round(float(np.degrees(scene["source_yaw"])) % 360, 1),
        "source_center_mm": [round(float(v) * 1000, 1) for v in scene["source_center"]],
    }


def goal_transform(seed, target="square", scene=None):
    """Goal centre and yaw for a seed (or for an explicit scene)."""
    scene = scene or layout(seed, target)
    return scene["goal_center"], scene["goal_yaw"]


def goal(seed, target="square", scene=None):
    from shapes import canonical

    outline, _, _ = canonical(target)
    center, yaw = goal_transform(seed, target, scene)
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
        (1 - 2 * (poses[..., 4] ** 2 + poses[..., 5] ** 2)) >= np.cos(np.deg2rad(FLAT_DEGREES)),
        axis=-1,
    )
    on_table = np.all(abs(poses[..., 2] - THICKNESS / 2) < TABLE_TOLERANCE, axis=-1)
    still = np.all(np.linalg.norm(velocities[..., :3], axis=-1) < 0.01, axis=-1)
    still &= np.all(np.linalg.norm(velocities[..., 3:], axis=-1) < 0.1, axis=-1)
    return {
        "iou": iou,
        "coverage": intersection / shapely.area(targets),
        "pieces_in_goal": (piece_coverage >= PIECE_COVERAGE).sum(axis=-1),
        "overlap_fraction": overlap,
        "on_table": on_table,
        "flat": flat,
        "still": still,
        "success": on_table
        & flat
        & still
        & (overlap <= OVERLAP_THRESHOLD)
        & (iou >= IOU_THRESHOLD)
        & (piece_coverage >= PIECE_COVERAGE).all(axis=-1),
    }


def score(poses, velocities, outline):
    """Convenient single-world version, also used by the geometry tests."""
    batch = score_batch(
        np.asarray(poses)[None], np.asarray(velocities)[None], np.asarray(outline)[None]
    )
    return {name: value[0].item() for name, value in batch.items()}
