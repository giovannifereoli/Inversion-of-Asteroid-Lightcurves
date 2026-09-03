"""Appendix B - gift wrapping, checked against Qhull."""

from __future__ import annotations

import numpy as np
import pytest

from lcinv import convex_hull, gift_wrap_hull, octant_triangulation
from lcinv.convexhull import ConvexHullError, hull_volume_area

CUBE = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def clouds():
    rng = np.random.default_rng(0)
    yield "gaussian", rng.normal(size=(200, 3))
    yield "uniform", rng.random((80, 3))
    yield "sphere-mesh", octant_triangulation(6).vertices
    yield "flattened", rng.normal(size=(300, 3)) * np.array([4.0, 1.0, 0.3])
    yield "cube+interior", np.vstack([CUBE, rng.uniform(-0.5, 0.5, (30, 3))])
    yield "box+interior", np.vstack([CUBE * [3.0, 1.0, 1.0], rng.uniform(-0.3, 0.3, (20, 3))])
    yield "coplanar-face-point", np.vstack([CUBE, [[0.0, 0.0, 1.0]], rng.uniform(-0.4, 0.4, (10, 3))])
    yield "duplicates", np.vstack([octant_triangulation(4).vertices] * 2)


@pytest.mark.parametrize("name,points", list(clouds()), ids=[n for n, _ in clouds()])
def test_giftwrap_matches_qhull(name, points):
    """Appendix B's N**2 method must agree exactly with the library route."""
    a = hull_volume_area(convex_hull(points, method="qhull"))
    b = hull_volume_area(gift_wrap_hull(points))
    assert b[0] == pytest.approx(a[0], rel=1e-9)
    assert b[1] == pytest.approx(a[1], rel=1e-9)


@pytest.mark.parametrize("name,points", list(clouds()), ids=[n for n, _ in clouds()])
def test_giftwrap_finds_the_same_hull_vertices(name, points):
    a = {tuple(p) for p in np.round(convex_hull(points, method="qhull").vertex_points(), 9)}
    b = {tuple(p) for p in np.round(gift_wrap_hull(points).vertex_points(), 9)}
    assert a == b


def test_all_points_lie_inside_the_hull():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(150, 3))
    hull = gift_wrap_hull(points)
    # Every plane must support the cloud: n . x + d <= 0 for all points.
    heights = points @ hull.equations[:, :3].T + hull.equations[:, 3]
    assert heights.max() < 1e-9


def test_cube_merges_into_six_square_faces():
    """Step 8: "remove those points that do not define new planes"."""
    polygons, normals, offsets = convex_hull(CUBE).merge_coplanar()
    assert len(polygons) == 6
    assert sorted(len(p) for p in polygons) == [4] * 6
    assert np.allclose(offsets, 1.0)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_merge_drops_points_interior_to_a_face():
    points = np.vstack([CUBE, [[0.0, 0.0, 1.0]]])
    polygons, _, _ = convex_hull(points).merge_coplanar()
    assert sorted(len(p) for p in polygons) == [4] * 6


def test_neighbour_lists_are_symmetric():
    hull = gift_wrap_hull(octant_triangulation(4).vertices)
    for vertex, neighbours in hull.neighbours.items():
        for other in neighbours:
            assert vertex in hull.neighbours[other]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        convex_hull(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        convex_hull(np.zeros((10, 3)), method="nope")


def test_degenerate_cloud_raises_convex_hull_error():
    flat = np.column_stack([np.random.default_rng(0).random((20, 2)), np.zeros(20)])
    with pytest.raises(ConvexHullError):
        convex_hull(flat, method="qhull")


def test_auto_uses_giftwrap_for_small_clouds():
    small = np.random.default_rng(1).normal(size=(50, 3))
    assert hull_volume_area(convex_hull(small, "auto"))[0] == pytest.approx(
        hull_volume_area(gift_wrap_hull(small))[0], rel=1e-12
    )
