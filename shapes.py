"""Small public silhouette corpus with explicit orientation-preserving certificates.

Coordinates use a 10 cm lattice. Certificates are rotations/translations of the
original seven tiles, never reflections. Public test figures are not secret.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from tangram import NOMINAL_VERTICES, SIDE, THICKNESS, TILES, VERTICES, rotation

VERSION = "silhouettes-v1"
PROMPT = "Assemble the tangram to match the silhouette using all seven pieces."
TARGETS = ("square", "rectangle", "house", "cat")
SUITES = {"train": ("square", "rectangle"), "dev": ("house",), "test": ("cat",)}
# Original, authored here. Tuple: quarter turns, x, y in the rotated lattice.
CERTIFICATES = {
    "rectangle": [(0, -2, 0), (1, 2, 0), (2, 5, -1), (3, 4, 4), (1, 2, 0), (0, 1, 2), (1, 2, 0)],
    "house": [(0, 0, 1), (0, 0, 1), (0, 1, 2), (2, 7, 0), (2, 3, 0), (0, -2, 1), (0, 1, 1)],
    "cat": [(0, -2, 0), (1, 2, 0), (3, 1, 3), (1, 0, 0), (3, 2, 4), (0, -2, 3), (1, 1, 2)],
}
OUTLINES = {
    "rectangle": [(0, 0), (4, 0), (4, 2), (0, 2)],
    "house": [(0, 0), (4, 0), (4, 1), (2, 3), (0, 1)],
    "cat": [(0, 0), (2, 0), (2, 4), (1, 3), (0, 4), (0, 2), (-1, 1)],
}


def canonical(name):
    """Return outline and one certificate (piece xy/yaw), in metres at the origin."""
    if name == "square":
        from tangram import CENTERS

        outline = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) - 0.5) * SIDE
        return outline, (CENTERS - 0.5) * SIDE, np.zeros(7)
    if name not in CERTIFICATES:
        raise ValueError(f"Target {name!r} is not defined; available: {', '.join(TARGETS)}")
    outline = np.array(OUTLINES[name], dtype=float) * 0.1
    center = (outline.min(axis=0) + outline.max(axis=0)) / 2
    outline -= center
    basis = np.array([[0.5, -0.5], [0.5, 0.5]])
    xy, yaws = [], []
    for tile, (quarter, x, y) in zip(TILES, CERTIFICATES[name]):
        polygon = np.array(tile) * 4 @ basis @ rotation(quarter * np.pi / 2).T + [x, y]
        xy.append(np.array(Polygon(polygon).centroid.coords[0]) * 0.1 - center)
        yaws.append(-np.pi / 4 + quarter * np.pi / 2)
    return outline, np.array(xy), np.array(yaws)


def solution(name, seed):
    """Privileged certificate for corpus verification/oracle baselines only."""
    from tangram import goal_transform

    _, xy, yaws = canonical(name)
    center, yaw = goal_transform(seed, name)
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
