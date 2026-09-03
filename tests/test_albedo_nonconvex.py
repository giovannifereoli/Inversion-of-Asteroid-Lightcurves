"""Section 3.3 (albedo) and Section 4 (nonconvex inversion)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import synthetic_set
from lcinv import (
    AlbedoSeparation,
    CylindricalShapeSeries,
    FacetGeometry,
    NonconvexInversion,
    Polyhedron,
    RadialShapeSeries,
    RayTracer,
    convexity_penalty,
    ellipsoid,
    octant_triangulation,
    paper_shape,
    sphere,
)
from lcinv.albedo import inverse_logistic_albedo, logistic_albedo
from lcinv.nonconvex import facet_radius_derivatives


class TestAlbedoSeparation:
    def test_logistic_stays_inside_the_interval(self):
        """Eq. (12)."""
        c = np.linspace(-50, 50, 201)
        w = logistic_albedo(c, 0.5, 1.5)
        assert w.min() >= 0.5 and w.max() <= 1.5
        assert logistic_albedo(0.0, 0.5, 1.5) == pytest.approx(1.0)

    def test_logistic_round_trip(self):
        w = np.array([0.6, 1.0, 1.4])
        assert logistic_albedo(inverse_logistic_albedo(w, 0.5, 1.5), 0.5, 1.5) == pytest.approx(w)

    def test_constant_albedo_produces_no_albedo_signal(self):
        """Section 3.5: "No albedo indication was obtained for surfaces with
        constant albedo"."""
        geometry = FacetGeometry.from_sphere(5)
        result = AlbedoSeparation(geometry, geometry.sphere_areas).run(max_nfev=400)
        assert np.ptp(result.albedo) < 1e-3
        assert result.facet_areas == pytest.approx(geometry.sphere_areas, rel=1e-3)
        assert result.residual_nonconvexity < 1e-3

    def test_a_bright_spot_shows_up_in_the_albedo(self):
        geometry = FacetGeometry.from_sphere(5)
        spot = geometry.normals @ np.array([0.0, 0.0, 1.0]) > 0.85
        albedo = np.where(spot, 1.4, 1.0)
        result = AlbedoSeparation(geometry, geometry.sphere_areas * albedo).run(max_nfev=600)
        # The paper is explicit that shape and albedo are "quantitatively
        # inseparable", so only the sign and location are checked.
        assert result.albedo[spot].mean() > result.albedo[~spot].mean()

    def test_area_part_is_pushed_towards_convexity(self):
        geometry = FacetGeometry.from_sphere(5)
        rng = np.random.default_rng(0)
        lumpy = geometry.sphere_areas * (1.0 + 0.3 * rng.random(len(geometry)))
        before = np.linalg.norm(lumpy @ geometry.normals) / lumpy.sum()
        result = AlbedoSeparation(geometry, lumpy, lambda_shape=10.0).run(max_nfev=600)
        assert result.residual_nonconvexity < before

    def test_rejects_bad_input(self):
        geometry = FacetGeometry.from_sphere(4)
        with pytest.raises(ValueError):
            AlbedoSeparation(geometry, np.ones(3))
        with pytest.raises(ValueError):
            AlbedoSeparation(geometry, geometry.sphere_areas, albedo_range=(1.5, 0.5))


class TestPaperDerivatives:
    """The formulas quoted in Section 4."""

    def test_area_and_mu_derivatives_match_finite_differences(self):
        mesh = octant_triangulation(4)
        rng = np.random.default_rng(0)
        radii = 1.0 + 0.25 * rng.normal(size=mesh.n_vertices)
        body = Polyhedron(mesh.vertices * radii[:, None], mesh.facets)
        earth = np.array([0.3, -0.5, 0.8])
        earth /= np.linalg.norm(earth)
        d_area, d_mu = facet_radius_derivatives(body, earth)

        step = 1e-7
        for facet in rng.choice(body.n_facets, 20, replace=False):
            for k in range(3):
                bumped = radii.copy()
                bumped[mesh.facets[facet, k]] += step
                other = Polyhedron(mesh.vertices * bumped[:, None], mesh.facets)
                fd_area = (other.areas[facet] - body.areas[facet]) / step
                fd_mu = (other.normals[facet] @ earth - body.normals[facet] @ earth) / step
                assert d_area[facet, k] == pytest.approx(fd_area, rel=1e-4, abs=1e-7)
                assert d_mu[facet, k] == pytest.approx(fd_mu, rel=1e-4, abs=1e-7)


class TestConvexityPenalty:
    def test_zero_for_a_convex_body(self):
        assert convexity_penalty(RayTracer(sphere(1.0, 6))) == 0.0
        assert convexity_penalty(RayTracer(ellipsoid(2.0, 1.2, 1.0, 6))) == 0.0

    def test_positive_and_ordered_for_concave_bodies(self):
        shallow = convexity_penalty(RayTracer(paper_shape("peanut", n_rows=6)))
        deep = convexity_penalty(RayTracer(paper_shape("binary", n_rows=6)))
        assert 0.0 < shallow < deep


class TestShapeSeries:
    def test_radial_series_reproduces_a_nonconvex_body(self):
        """Eq. (15) with the paper's degree-4 truncation."""
        target = paper_shape("peanut", n_rows=6)
        series = RadialShapeSeries(n_rows=6, lmax=4)
        assert series.n_parameters == 25
        body = series.body(series.fit(target))
        assert body.volume == pytest.approx(target.volume, rel=0.12)
        assert 1 - body.volume / body.convex_hull().volume > 0.05

    def test_cylindrical_series_builds_a_closed_surface(self):
        """Eq. (16)."""
        target = paper_shape("castalia", n_rows=7)
        series = CylindricalShapeSeries.from_body(target, n_x=14, n_phi=20)
        body = series.body(series.fit(target))
        assert np.linalg.norm(body.facet_normal_sum) < 1e-9 * body.surface_area
        assert body.volume > 0
        assert body.volume == pytest.approx(target.volume, rel=0.15)

    def test_uniform_coefficients_give_a_cylinder(self):
        series = CylindricalShapeSeries(n_x=10, n_phi=24, degree_x=0, degree_phi=0, half_length=1.0)
        body = series.body(np.array([np.log(0.5)]))
        assert np.linalg.norm(body.facet_normal_sum) < 1e-12
        # Two cones plus a barrel, all of radius 0.5.
        assert body.volume < np.pi * 0.25 * 2.0
        assert body.volume > 0.5 * np.pi * 0.25 * 2.0

    def test_extreme_coefficients_do_not_produce_infinities(self):
        series = RadialShapeSeries(n_rows=4, lmax=2)
        body = series.body(np.full(series.n_parameters, 50.0))
        assert np.isfinite(body.vertices).all()


class TestScaleDegeneracy:
    """Eq. (13) is scale-free, so the series' constant term must be frozen.

    Section 3.4 says so for the convex case - "the coefficient a00 in (8) is a
    scale factor as well, so it can be left out of the parameter set" - and the
    constant term of Eq. (15) plays exactly that role.  Left free, the
    optimiser slides down the flat direction until the body collapses.
    """

    def test_relative_objective_freezes_the_scale(self, spin, law):
        data = synthetic_set(paper_shape("peanut", n_rows=6), spin, law, n_curves=3, n_points=15)
        series = RadialShapeSeries(n_rows=4, lmax=2)
        inv = NonconvexInversion(data, spin, series=series, law=law)
        assert inv.fix_scale is True
        inv._fixed_scale = 0.25
        shifted = np.zeros(series.n_parameters)
        shifted[series.scale_index] = -12.0  # would shrink the body enormously
        assert inv._apply_scale(shifted)[series.scale_index] == 0.25

    def test_absolute_objective_leaves_the_scale_free(self, spin, law):
        data = synthetic_set(paper_shape("peanut", n_rows=6), spin, law, n_curves=3, n_points=15)
        inv = NonconvexInversion(
            data, spin, series=RadialShapeSeries(n_rows=4, lmax=2), law=law,
            objective="absolute",
        )
        assert inv.fix_scale is False

    def test_body_does_not_collapse_during_a_fit(self, spin, law):
        data = synthetic_set(paper_shape("peanut", n_rows=6), spin, law, n_curves=4, n_points=20)
        series = RadialShapeSeries(n_rows=4, lmax=2)
        inv = NonconvexInversion(data, spin, series=series, law=law)
        result = inv.run(initial=series.fit(paper_shape("peanut", n_rows=6)), max_nfev=80)
        radii = np.linalg.norm(result.body.vertices, axis=1)
        assert radii.min() > 1e-3
        assert result.body.volume > 1e-3
        # A star-shaped body can never exceed its own convex hull.
        assert result.body.volume <= result.body.convex_hull().volume * (1 + 1e-9)


@pytest.mark.slow
class TestNonconvexInversion:
    def test_improves_on_a_convex_starting_guess(self, spin, law):
        """Section 4 wants the iteration started from the convex result."""
        truth = paper_shape("peanut", n_rows=6)
        data = synthetic_set(truth, spin, law, n_curves=6, n_points=25, seed=7)
        series = RadialShapeSeries(n_rows=5, lmax=2)
        inv = NonconvexInversion(data, spin, series=series, law=law)

        start = series.fit(truth.convex_hull())
        before = float(np.sum(inv._residuals(start) ** 2))
        result = inv.run(initial=start, max_nfev=120)
        assert result.chi2 <= before
        assert result.body.n_facets == 8 * 5**2

    def test_recovers_a_known_concavity(self, spin, law):
        """Section 4: "a conception of the largest nonconvexities can be
        formed", though "the depths of the valleys are not very precise"."""
        truth = paper_shape("peanut", n_rows=7)
        data = synthetic_set(truth, spin, law, n_curves=10, n_points=40, seed=21)
        series = RadialShapeSeries(n_rows=6, lmax=3)
        inv = NonconvexInversion(data, spin, series=series, law=law)
        start = series.fit(truth.convex_hull())
        result = inv.run(initial=start, max_nfev=300)

        assert result.chi2 < 0.6 * float(np.sum(inv._residuals(start) ** 2))
        depth = 1 - result.body.volume / result.body.convex_hull().volume
        true_depth = 1 - truth.volume / truth.convex_hull().volume
        # The concavity is found, but shallower than the truth.
        assert 0.3 * true_depth < depth < true_depth
        assert result.convexity_penalty > 0.0

    def test_regularisation_penalises_concavity(self, spin, law):
        truth = paper_shape("peanut", n_rows=6)
        data = synthetic_set(truth, spin, law, n_curves=4, n_points=20, seed=8)
        series = RadialShapeSeries(n_rows=5, lmax=2)
        coeffs = series.fit(truth)
        plain = NonconvexInversion(data, spin, series=series, law=law, regularisation=0.0)
        regularised = NonconvexInversion(data, spin, series=series, law=law, regularisation=10.0)
        assert len(regularised._residuals(coeffs)) == len(plain._residuals(coeffs)) + 1
        assert np.sum(regularised._residuals(coeffs) ** 2) > np.sum(plain._residuals(coeffs) ** 2)
