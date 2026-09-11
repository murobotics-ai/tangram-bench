"""Small public silhouette corpus with explicit orientation-preserving certificates.

Coordinates are in 10 cm units of each figure's upright frame; `L(a, b)` is
a + b*sqrt(2), the two lengths a tangram needs. A certificate lists, per piece
in tile order, its rotation in eighth turns and where its first nominal vertex
lands. Rotations and translations only, never reflections. The public figures
are not secret.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from tangram import NOMINAL_VERTICES, SIDE, THICKNESS, TILES, VERTICES, rotation

VERSION = "silhouettes"
TARGETS = ("square", "rectangle", "house", "cat")
# The general task prompt names the figure; per-step subtasks are recorded by
# demonstrators (see examples/oracle.py) and never shown to a policy at evaluation.
PROMPT = "Solve the tangram puzzle to assemble the {target}."
# Three training figures and one held-out figure. "dev" evaluates the training
# figures on unseen seeds (unseen poses); "test" is the unseen silhouette.
SUITES = {
    "train": ("square", "rectangle", "house"),
    "dev": ("square", "rectangle", "house"),
    "test": ("cat",),
}


def L(a, b=0):
    return a + b * np.sqrt(2)


def prompt(target):
    if target not in TARGETS:
        raise ValueError(f"Target {target!r} is not defined; available: {', '.join(TARGETS)}")
    return PROMPT.format(target=target)


# Outlines are counterclockwise. The house is the classic tangram house (body,
# overhanging roof, chimney), mirrored so the parallelogram needs no flip.
OUTLINES = {
    "square": [(0, 0), (L(0, 2), 0), (L(0, 2), L(0, 2)), (0, L(0, 2))],
    "rectangle": [(0, 0), (4, 0), (4, 2), (0, 2)],
    "house": [
        (0, 0),
        (L(0, 2), 0),
        (L(0, 2), L(0, 1)),
        (L(0.5, 2), L(0, 1)),
        (L(-0.5, 2), L(1, 1)),
        (L(-0.5, 2), L(2, 1)),
        (L(-1.5, 2), L(2, 1)),
        (L(-1.5, 2), L(1, 1)),
        (L(-0.5, 1), L(0, 2)),
        (-0.5, L(0, 1)),
        (0, L(0, 1)),
    ],
    "cat": [(0, 0), (2, 0), (2, 4), (1, 3), (0, 4), (0, 2), (-1, 1)],
}
# Original, authored here. Tuple per piece: eighth turns, x, y of its first vertex.
CERTIFICATES = {
    "square": [(0, L(0, 2 * x), L(0, 2 * y)) for (x, y), *_ in TILES],
    "rectangle": [(7, 0, 2), (1, 2, 0), (3, 3, 1), (5, 4, 0), (1, 2, 1), (7, 3, 2), (1, 2, 0)],
    "house": [
        (4, L(-0.5, 2), L(0, 1)),
        (2, L(0, 2), 0),
        (2, L(0, 2), L(0, 1)),
        (2, 0, L(0, 1)),
        (6, 0, L(0, 1)),
        (1, L(-1.5, 2), L(1, 1)),
        (3, L(0.5, 2), L(0, 1)),
    ],
    "cat": [(7, 0, 2), (1, 2, 0), (5, -1, 1), (1, 0, 4), (5, 2, 3), (7, 0, 3), (1, 1, 2)],
}


def canonical(name):
    """Return outline and one certificate (piece xy/yaw), in metres at the origin."""
    if name not in CERTIFICATES:
        raise ValueError(f"Target {name!r} is not defined; available: {', '.join(TARGETS)}")
    outline = np.array(OUTLINES[name], dtype=float) * 0.1
    center = (outline.min(axis=0) + outline.max(axis=0)) / 2
    outline -= center
    xy, yaws = [], []
    for vertices, (eighths, x, y) in zip(NOMINAL_VERTICES, CERTIFICATES[name]):
        yaw = eighths * np.pi / 4
        xy.append(np.array([x, y]) * 0.1 - rotation(yaw) @ vertices[0] - center)
        yaws.append(yaw)
    return outline, np.array(xy), np.array(yaws)


def solution(name, seed, scene=None):
    """Privileged certificate for corpus verification/oracle baselines only."""
    from tangram import goal_transform

    _, xy, yaws = canonical(name)
    center, yaw = goal_transform(seed, name, scene)
    poses = np.zeros((7, 7))
    poses[:, :2] = xy @ rotation(yaw).T + center
    poses[:, 2] = THICKNESS / 2
    poses[:, 3], poses[:, 6] = np.cos((yaws + yaw) / 2), np.sin((yaws + yaw) / 2)
    return poses


def validate(name):
    """Exact nominal coverage and physical inset ceiling; independent of rendering."""
    outline, xy, yaws = canonical(name)
    nominal = [Polygon(v @ rotation(a).T + p) for v, p, a in zip(NOMINAL_VERTICES, xy, yaws)]
    target, union = Polygon(outline), unary_union(nominal)
    if not target.is_valid or target.interiors or abs(target.area - SIDE**2) > 1e-10:
        raise ValueError(f"Invalid silhouette: {name}")
    if union.symmetric_difference(target).area > 1e-10:
        raise ValueError(f"Certificate does not cover {name}")
    if sum(p.area for p in nominal) - union.area > 1e-10:
        raise ValueError(f"Certificate overlaps: {name}")
    physical = unary_union([Polygon(v @ rotation(a).T + p) for v, p, a in zip(VERTICES, xy, yaws)])
    ceiling = physical.intersection(target).area / physical.union(target).area
    if ceiling < 0.95:
        raise ValueError(f"Unachievable success threshold: {name}")
    return {"target": name, "area": target.area, "iou_ceiling": ceiling, "pieces": 7}


if __name__ == "__main__":
    import json

    print(json.dumps([validate(name) for name in TARGETS], indent=2))
