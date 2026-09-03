"""Appendix C - Minkowski minimisation."""

from __future__ import annotations

import numpy as np
import pytest

from lcinv import ellipsoid, minkowski_solve, sphere
from lcinv.minkowski import close_facet_areas, dual_polyhedron, merge_duplicate_normals

CUBE_NORMALS = np.array(
    [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
)


class TestDualTransform:
    def test_unit_distances_give_the_unit_cube(self):
        """Eq. (19), r = n / l, maps the six planes to a 2x2x2 cube."""
        body, areas = dual_polyhedron(CUBE_NORMALS, np.ones(6))
        assert body.volume == pytest.approx(8.0)
        assert areas == pytest.approx(np.full(6, 4.0))

    def test_distances_set_the_box_dimensions(self):
        body, areas = dual_polyhedron(CUBE_NORMALS, np.array([3.0, 3, 2, 2, 1, 1]))
        assert body.volume == pytest.approx(6 * 4 * 2)
        assert body.extents() == pytest.approx([6.0, 4.0, 2.0])

    def test_rejects_non_positive_distances(self):
        with pytest.raises(ValueError):
            dual_polyhedron(CUBE_NORMALS, np.array([1.0, 1, 1, 1, 1, -1]))

    def test_unused_normals_get_zero_area(self):
        normals = np.vstack([CUBE_NORMALS, [[0.577, 0.577, 0.577]]])
        # A far-away plane cannot cut the cube.
        _, areas = dual_polyhedron(normals, np.array([1.0, 1, 1, 1, 1, 1, 50.0]))
        assert areas[-1] == 0.0


class TestClosure:
    def test_dark_facet_makes_equation_3_vanish(self):
        """Section 3.3's "adding a facet of corresponding size"."""
        normals = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
        n2, g2 = close_facet_areas(normals, np.ones(3))
        assert len(g2) == 4
        assert np.linalg.norm(g2 @ n2) < 1e-12

    def test_already_closed_input_is_untouched(self):
        body = sphere(1.0, 6)
        n2, g2 = close_facet_areas(body.normals, body.areas)
        assert len(g2) == body.n_facets

    def test_duplicate_normals_are_summed(self):
        normals = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1, 0]])
        n, g, inverse = merge_duplicate_normals(normals, np.array([1.0, 2.0, 5.0]))
        assert len(n) == 2
        assert sorted(g) == [3.0, 5.0]
        assert len(inverse) == 3


class TestReconstruction:
    @pytest.mark.parametrize(
        "body,label",
        [(sphere(1.0, 6), "sphere"), (ellipsoid(2.0, 1.3, 1.0, 6), "ellipsoid")],
    )
    def test_recovers_a_body_from_its_own_facet_areas(self, body, label):
        result = minkowski_solve(body.normals, body.areas, max_iter=200)
        assert result.converged
        assert result.alignment > 1 - 1e-7
        assert result.volume == pytest.approx(body.volume, rel=1e-3)
        assert result.polyhedron.extents() == pytest.approx(body.extents(), rel=5e-3)

    def test_realised_areas_match_the_request(self):
        body = ellipsoid(1.6, 1.2, 1.0, 6)
        result = minkowski_solve(body.normals, body.areas, max_iter=200)
        assert len(result.areas) == body.n_facets
        assert result.areas == pytest.approx(body.areas, rel=0.03)
        assert result.dark_area < 1e-3 * body.surface_area

    def test_areas_are_aligned_with_the_input_normals(self):
        """A permutation of the inputs must permute the outputs the same way."""
        body = ellipsoid(1.6, 1.2, 1.0, 5)
        order = np.random.default_rng(0).permutation(body.n_facets)
        plain = minkowski_solve(body.normals, body.areas, max_iter=120)
        shuffled = minkowski_solve(body.normals[order], body.areas[order], max_iter=120)
        assert shuffled.areas == pytest.approx(plain.areas[order], rel=1e-6, abs=1e-12)

    def test_the_scale_convention_reproduces_the_unit_sphere(self):
        """Appendix C's final scaling, in the corrected direction."""
        body = sphere(1.0, 6)
        result = minkowski_solve(body.normals, body.areas, max_iter=200)
        radii = np.linalg.norm(result.polyhedron.vertices, axis=1)
        assert radii.mean() == pytest.approx(1.0, rel=5e-3)

    def test_box_from_prescribed_areas(self):
        result = minkowski_solve(CUBE_NORMALS, np.array([2.0, 2, 3, 3, 6, 6]))
        assert result.volume == pytest.approx(6.0, rel=1e-4)
        assert result.polyhedron.extents() == pytest.approx([3.0, 2.0, 1.0], rel=1e-3)

    def test_result_is_centred(self):
        body = ellipsoid(1.8, 1.2, 1.0, 5)
        result = minkowski_solve(body.normals, body.areas, max_iter=120)
        assert result.polyhedron.centroid == pytest.approx(np.zeros(3), abs=1e-6)

    def test_scaling_the_areas_scales_the_body(self):
        body = sphere(1.0, 5)
        a = minkowski_solve(body.normals, body.areas, max_iter=120)
        b = minkowski_solve(body.normals, 4.0 * body.areas, max_iter=120)
        assert b.volume == pytest.approx(8.0 * a.volume, rel=5e-3)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            minkowski_solve(np.zeros((4, 2)), np.ones(4))
        with pytest.raises(ValueError):
            minkowski_solve(CUBE_NORMALS, np.ones(5))
        with pytest.raises(ValueError):
            minkowski_solve(CUBE_NORMALS, -np.ones(6))
