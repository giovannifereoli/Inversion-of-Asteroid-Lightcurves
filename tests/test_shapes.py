"""The Section 3.5 test bodies."""

from __future__ import annotations

import numpy as np
import pytest

from lcinv import (
    PAPER_SHAPE_NAMES,
    Polyhedron,
    binary,
    castalia_like,
    ellipsoid,
    gaussian_random_sphere,
    irregular_shape,
    paper_shape,
    peanut,
    sphere,
)
from lcinv.shapes import radial_body


@pytest.mark.parametrize("which", [1, 2, 3, 4] + list(PAPER_SHAPE_NAMES))
def test_paper_shapes_are_valid_closed_bodies(which):
    body = paper_shape(which)
    assert body.volume > 0
    assert np.linalg.norm(body.facet_normal_sum) < 1e-9 * body.surface_area


def test_nonconvexity_increases_through_the_series():
    """"Four examples of increasing nonconvexity"."""
    ratios = [
        1.0 - paper_shape(name).volume / paper_shape(name).convex_hull().volume
        for name in PAPER_SHAPE_NAMES
    ]
    assert ratios[0] < ratios[1] < ratios[3]
    assert ratios[3] > 0.2  # the binary is by far the most nonconvex


def test_binary_has_two_separate_components():
    body = binary(n_rows=5)
    # Two disjoint closed surfaces: the hull is much larger than the body.
    assert body.volume < 0.7 * body.convex_hull().volume
    assert body.extents()[0] > 2.0 * body.extents()[1]


def test_peanut_has_a_waist():
    body = peanut(n_rows=7)
    near_waist = np.abs(body.vertices[:, 0]) < 0.1
    far = np.abs(body.vertices[:, 0]) > 0.5
    waist_radius = np.linalg.norm(body.vertices[near_waist][:, 1:], axis=1).max()
    body_radius = np.linalg.norm(body.vertices[far][:, 1:], axis=1).max()
    assert waist_radius < 0.75 * body_radius


def test_castalia_is_bilobed():
    body = castalia_like(n_rows=8)
    assert body.extents()[0] > 1.8 * body.extents()[2]
    assert 1.0 - body.volume / body.convex_hull().volume > 0.05


def test_irregular_shape_is_almost_convex():
    body = irregular_shape()
    assert 1.0 - body.volume / body.convex_hull().volume < 0.02


def test_shape_selection_by_name_and_number_agree():
    for i, name in enumerate(PAPER_SHAPE_NAMES, 1):
        assert paper_shape(i).volume == pytest.approx(paper_shape(name).volume)


def test_rejects_unknown_shapes():
    with pytest.raises(ValueError):
        paper_shape("teapot")
    with pytest.raises(ValueError):
        paper_shape(9)


def test_gaussian_random_sphere_has_the_requested_scatter():
    body = gaussian_random_sphere(sigma=0.2, nu=3.0, lmax=6, n_rows=8, seed=1)
    radii = np.linalg.norm(body.vertices, axis=1)
    assert radii.std() / radii.mean() == pytest.approx(0.2, rel=0.1)
    assert body.volume > 0


def test_gaussian_random_sphere_is_reproducible():
    a = gaussian_random_sphere(seed=3)
    b = gaussian_random_sphere(seed=3)
    assert a.vertices == pytest.approx(b.vertices)


def test_gaussian_random_sphere_rejects_bad_sigma():
    with pytest.raises(ValueError):
        gaussian_random_sphere(sigma=0.0)


def test_radial_body_builds_a_sphere():
    body = radial_body(lambda theta, phi: np.full_like(theta, 2.0), n_rows=8)
    assert np.allclose(np.linalg.norm(body.vertices, axis=1), 2.0)


def test_radial_body_rejects_bad_radii():
    with pytest.raises(ValueError):
        radial_body(lambda theta, phi: np.zeros_like(theta), n_rows=4)
    with pytest.raises(ValueError):
        radial_body(lambda theta, phi: np.ones(3), n_rows=4)


def test_sphere_and_ellipsoid_helpers():
    assert isinstance(sphere(2.0, 5), Polyhedron)
    assert ellipsoid(2.0, 1.0, 1.0, 12).volume == pytest.approx(
        2.0 * sphere(1.0, 12).volume, rel=1e-9
    )
