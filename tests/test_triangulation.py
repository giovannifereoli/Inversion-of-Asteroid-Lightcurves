"""Appendix A - the octant triangulation."""

from __future__ import annotations

import numpy as np
import pytest

from lcinv import Polyhedron, octant_triangulation
from lcinv.triangulation import facet_adjacency, fibonacci_directions, triangulated_ellipsoid


@pytest.mark.parametrize("n", [1, 2, 4, 8, 11])
def test_counts_match_appendix_a(n):
    """"The number of facets is 8 N**2, and the number of vertices 4 N**2 + 2"."""
    mesh = octant_triangulation(n)
    assert mesh.n_facets == 8 * n**2
    assert mesh.n_vertices == 4 * n**2 + 2


@pytest.mark.parametrize("n", [3, 6])
def test_vertices_lie_on_the_unit_sphere(n):
    mesh = octant_triangulation(n)
    assert np.allclose(np.linalg.norm(mesh.vertices, axis=1), 1.0)


@pytest.mark.parametrize("n", [2, 5, 8])
def test_surface_is_closed_and_outward(n):
    body = Polyhedron(*astuple(octant_triangulation(n)))
    assert body.volume > 0
    # A closed surface satisfies sum_j A_j n_j = 0, the identity behind Eq. (3).
    assert np.linalg.norm(body.facet_normal_sum) < 1e-12 * body.surface_area


def astuple(mesh):
    return mesh.vertices, mesh.facets


@pytest.mark.parametrize("n", [4, 8, 16])
def test_converges_to_the_sphere(n):
    body = Polyhedron(*astuple(octant_triangulation(n)))
    # An inscribed polyhedron approaches 4 pi / 3 from below as O(1/N^2).
    assert body.volume < 4.0 * np.pi / 3.0
    assert body.volume > 4.0 * np.pi / 3.0 * (1.0 - 6.0 / n**2)


def test_every_facet_has_three_edge_neighbours():
    adjacency = facet_adjacency(octant_triangulation(5).facets)
    assert all(len(n) == 3 for n in adjacency)
    for i, neighbours in enumerate(adjacency):
        for j in neighbours:
            assert i in adjacency[j]


def test_polar_facets_are_comparable_to_equatorial_ones():
    """Appendix A: "the polar facets are about the same size as the equatorial
    ones for a roughly spherical body"."""
    body = Polyhedron(*astuple(octant_triangulation(8)))
    z = np.abs(body.facet_centroids[:, 2]) / np.linalg.norm(body.facet_centroids, axis=1)
    polar = body.areas[z > 0.9].mean()
    equatorial = body.areas[z < 0.2].mean()
    assert 0.5 < polar / equatorial < 2.0


def test_rejects_bad_row_counts():
    with pytest.raises(ValueError):
        octant_triangulation(0)


def test_fibonacci_directions_are_unit_and_spread():
    dirs = fibonacci_directions(500)
    assert dirs.shape == (500, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert np.linalg.norm(dirs.mean(axis=0)) < 0.05


def test_triangulated_ellipsoid_has_the_right_volume():
    body = Polyhedron(*triangulated_ellipsoid(14, (2.0, 1.0, 0.5)))
    assert body.volume == pytest.approx(4.0 / 3.0 * np.pi * 1.0, rel=0.02)
