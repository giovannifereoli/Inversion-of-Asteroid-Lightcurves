"""Section 3 - convex inversion, including a check against the reference C code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_set
from lcinv import (
    FacetGeometry,
    FacetInversion,
    HarmonicInversion,
    LightcurveSet,
    LommelSeeligerLambert,
    Objective,
    PhaseFunction,
    SpinState,
    ellipsoid,
    nonconvexity_residual,
    paper_shape,
)
from lcinv.convex import ConvexModel, ellipsoid_log_curvature

REFERENCE = Path(__file__).parent / "data" / "test_lcs_rel.txt"


class TestForwardModel:
    def test_design_matrix_vanishes_out_of_view(self):
        """Eq. (4)."""
        geometry = FacetGeometry.from_sphere(4)
        model = ConvexModel(geometry, LommelSeeligerLambert(1.0))
        earth = np.array([[0.0, 0.0, 1.0]])
        design = model.design_matrix(earth, earth)
        assert np.all(design[0][geometry.normals[:, 2] <= 0] == 0.0)
        assert np.all(design[0][geometry.normals[:, 2] > 0] > 0.0)

    def test_brightness_is_the_matrix_product(self):
        geometry = FacetGeometry.from_sphere(4)
        model = ConvexModel(geometry, LommelSeeligerLambert(1.0))
        rng = np.random.default_rng(0)
        dirs = rng.normal(size=(5, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        design = model.design_matrix(dirs, dirs)
        g = rng.random(len(geometry))
        assert model.brightness(g, design) == pytest.approx(design @ g)

    def test_ellipsoid_curvature_function_integrates_to_its_area(self):
        """G(n) = (abc)^2 / (a^2 nx^2 + b^2 ny^2 + c^2 nz^2)^2, and Eq. (10)
        turns it into facet areas summing to the surface area."""
        geometry = FacetGeometry.from_sphere(20)
        a, b, c = 2.0, 1.4, 1.0
        g = np.exp(ellipsoid_log_curvature(geometry, a, b, c)) * geometry.sphere_areas
        assert g.sum() == pytest.approx(ellipsoid(a, b, c, 20).surface_area, rel=5e-3)

    def test_curvature_of_a_sphere_is_constant(self):
        geometry = FacetGeometry.from_sphere(6)
        assert np.ptp(ellipsoid_log_curvature(geometry, 1.0, 1.0, 1.0)) < 1e-12


class TestJacobians:
    """The analytic derivatives must match finite differences."""

    @pytest.mark.parametrize(
        "objective", [Objective.RELATIVE, Objective.RENORMALISED, Objective.ABSOLUTE]
    )
    def test_harmonic_jacobian(self, ellipsoid_data, spin, law, objective):
        geometry = FacetGeometry.from_sphere(5)
        inv = HarmonicInversion(
            ellipsoid_data, geometry, spin, lmax=3, law=law,
            objective=objective, convexity_weight=0.3,
        )
        p = inv.initial_coefficients(1.4, 1.0, 0.9)
        inv._fixed_a00 = float(p[0])
        analytic = inv._jacobian_fn(p)
        base = inv._residual_fn(p)
        for k in (1, 5, 9):
            step = 1e-6
            bumped = p.copy()
            bumped[k] += step
            numeric = (inv._residual_fn(bumped) - base) / step
            assert analytic[:, k] == pytest.approx(numeric, rel=2e-4, abs=1e-8)

    def test_facet_gradient(self, ellipsoid_data, spin, law):
        geometry = FacetGeometry.from_sphere(4)
        inv = FacetInversion(
            ellipsoid_data, geometry, spin, law=law, convexity_weight=0.3
        )
        rng = np.random.default_rng(0)
        a = np.log(geometry.sphere_areas) + 0.05 * rng.normal(size=len(geometry))
        value, grad = inv._objective_and_gradient(a)
        for k in rng.choice(len(a), 6, replace=False):
            step = 1e-7
            bumped = a.copy()
            bumped[k] += step
            numeric = (inv._objective_and_gradient(bumped)[0] - value) / step
            assert grad[k] == pytest.approx(numeric, rel=1e-3, abs=1e-9)


class TestObjectives:
    def test_relative_objective_ignores_a_rescaled_model(self, ellipsoid_data, spin, law):
        """Eq. (13) "discards all scale factors"."""
        geometry = FacetGeometry.from_sphere(5)
        inv = HarmonicInversion(
            ellipsoid_data, geometry, spin, lmax=3, law=law,
            objective=Objective.RELATIVE, convexity_weight=0.0,
            convexity_components="none",
        )
        p = inv.initial_coefficients(1.4, 1.0, 0.9)
        inv._fixed_a00 = float(p[0])
        scaled = p.copy()
        scaled[0] += np.log(7.3)  # multiplies every g_j by 7.3
        inv.fix_scale = False
        assert inv._residual_fn(scaled) == pytest.approx(inv._residual_fn(p))

    def test_absolute_objective_does_not(self, ellipsoid_data, spin, law):
        geometry = FacetGeometry.from_sphere(5)
        inv = HarmonicInversion(
            ellipsoid_data, geometry, spin, lmax=3, law=law,
            objective=Objective.ABSOLUTE, convexity_weight=0.0,
            convexity_components="none",
        )
        p = inv.initial_coefficients(1.4, 1.0, 0.9)
        scaled = p.copy()
        scaled[0] += np.log(2.0)
        assert not np.allclose(inv._residual_fn(scaled), inv._residual_fn(p))

    def test_convexity_rows_are_equation_three(self, ellipsoid_data, spin, law):
        geometry = FacetGeometry.from_sphere(4)
        inv = FacetInversion(
            ellipsoid_data, geometry, spin, law=law, convexity_weight=2.0
        )
        g = np.random.default_rng(0).random(len(geometry))
        rows, _ = inv._convexity_rows(g)
        assert rows == pytest.approx(2.0 * (g @ geometry.normals))

    def test_z_only_regularisation_keeps_one_row(self, ellipsoid_data, spin, law):
        """Section 3.4: "it is sufficient to include only the z-component"."""
        geometry = FacetGeometry.from_sphere(4)
        inv = FacetInversion(
            ellipsoid_data, geometry, spin, law=law,
            convexity_weight=1.0, convexity_components="z",
        )
        rows, jac = inv._convexity_rows(np.ones(len(geometry)))
        assert rows.shape == (1,)
        assert jac.shape == (1, len(geometry))

    def test_none_disables_regularisation(self, ellipsoid_data, spin, law):
        geometry = FacetGeometry.from_sphere(4)
        inv = FacetInversion(
            ellipsoid_data, geometry, spin, law=law, convexity_components="none"
        )
        rows, _ = inv._convexity_rows(np.ones(len(geometry)))
        assert rows.shape == (0,)

    def test_rejects_unknown_settings(self, ellipsoid_data, spin, law):
        geometry = FacetGeometry.from_sphere(4)
        with pytest.raises(ValueError):
            FacetInversion(ellipsoid_data, geometry, spin, law=law, convexity_components="q")
        with pytest.raises(ValueError):
            FacetInversion(LightcurveSet([]), geometry, spin, law=law)


class TestRecovery:
    def test_recovers_a_triaxial_ellipsoid(self, spin, law):
        """The cleanest end-to-end check: a convex body with no shadowing."""
        truth = ellipsoid(1.6, 1.15, 0.95, 8)
        data = synthetic_set(truth, spin, law, n_curves=10, seed=2)
        geometry = FacetGeometry.from_sphere(7)
        result = HarmonicInversion(data, geometry, spin, lmax=6, law=law).run()
        assert result.rms < 5e-3
        # A constant-albedo convex body must come out convex: Section 3.5 quotes
        # 0.001-0.007 for the residual nonconvexity ratio.
        assert result.nonconvexity < 0.02
        shape = result.shape(geometry, max_iter=150)
        ratios = shape.polyhedron.extents() / shape.polyhedron.extents()[2]
        assert ratios[0] == pytest.approx(1.6 / 0.95, rel=0.10)
        assert ratios[1] == pytest.approx(1.15 / 0.95, rel=0.10)

    def test_noise_does_not_break_the_solution(self, spin, law):
        """Section 3.5: 5-10% noise "does not cause a need for regularization"."""
        truth = ellipsoid(1.6, 1.15, 0.95, 8)
        clean = synthetic_set(truth, spin, law, n_curves=10, seed=3)
        noisy = clean.with_noise(0.05, seed=3)
        geometry = FacetGeometry.from_sphere(7)
        result = HarmonicInversion(noisy, geometry, spin, lmax=6, law=law).run()
        assert 0.02 < result.rms < 0.12
        assert result.nonconvexity < 0.05

    def test_facet_method_improves_on_the_series(self, spin, law):
        """Section 3.5: the series gives "a fast initial solution that can then
        be enhanced with the polyhedron method"."""
        data = synthetic_set(paper_shape("peanut", n_rows=7), spin, law, n_curves=10, seed=4)
        geometry = FacetGeometry.from_sphere(7)
        series = HarmonicInversion(data, geometry, spin, lmax=6, law=law).run()
        facet = FacetInversion(data, geometry, spin, law=law).run(
            initial=series.areas, max_iter=400
        )
        assert facet.chi2 < series.chi2

    def test_inversion_of_a_nonconvex_body_approaches_its_hull(self, spin, law):
        """Section 3.5: results "very close to the convex hulls of the original
        bodies"."""
        truth = paper_shape("peanut", n_rows=7)
        data = synthetic_set(truth, spin, law, n_curves=12, seed=5)
        geometry = FacetGeometry.from_sphere(7)
        result = HarmonicInversion(data, geometry, spin, lmax=6, law=law).run()
        shape = result.shape(geometry, max_iter=150)
        hull = truth.convex_hull()
        got = shape.polyhedron.extents() / shape.polyhedron.extents()[2]
        want = hull.extents() / hull.extents()[2]
        # The paper notes the long axis is contracted for strongly nonconvex
        # bodies, so this is a loose but meaningful bound.
        assert got[0] == pytest.approx(want[0], rel=0.30)
        assert got[1] == pytest.approx(want[1], rel=0.30)

    def test_nonconvexity_residual_is_small_for_constant_albedo(self, spin, law):
        truth = ellipsoid(1.5, 1.1, 1.0, 7)
        data = synthetic_set(truth, spin, law, n_curves=10, seed=6)
        geometry = FacetGeometry.from_sphere(6)
        result = HarmonicInversion(data, geometry, spin, lmax=5, law=law).run()
        ratio = np.linalg.norm(
            nonconvexity_residual(geometry, result.areas)
        ) / result.areas.sum()
        assert ratio == pytest.approx(result.nonconvexity)
        assert ratio < 0.02


@pytest.mark.slow
class TestAgainstReferenceImplementation:
    """Cross-check against Durech's C `convexinv`, distributed by DAMIT.

    Running that program on this exact dataset with its shipped
    `input_convexinv` (lambda 220, beta 0, P 5.76198 free; convexity 0.1;
    harmonics 6x6; 8 rows; LSL with a=0.5, d=0.1, k=-0.5, c=0.1) stops after
    its 50-iteration cap at chi2 = 0.378227, dev = 0.015667, and reports
    lambda = 253.308141, beta = -16.917887, P = 5.761982.
    """

    REF_CHI2 = 0.378227
    REF_LAMBDA, REF_BETA, REF_PERIOD = 253.308141, -16.917887, 5.761982

    @pytest.fixture(scope="class")
    def setup(self):
        data = LightcurveSet.from_damit_txt(REFERENCE)
        t0 = float(int(min(c.jd.min() for c in data)))
        law = LommelSeeligerLambert(0.1, PhaseFunction(0.5, 0.1, -0.5))
        return data, t0, law, FacetGeometry.from_sphere(8)

    def test_dataset_is_the_expected_one(self, setup):
        data, *_ = setup
        assert len(data) == 37
        assert data.n_points == 1541
        assert not data.all_calibrated

    def test_matches_the_reference_at_the_reference_spin(self, setup):
        data, t0, law, geometry = setup
        spin = SpinState(self.REF_LAMBDA, self.REF_BETA, self.REF_PERIOD, t0, 0.0)
        result = HarmonicInversion(
            data, geometry, spin, lmax=6, law=law,
            objective=Objective.RELATIVE, convexity_weight=0.1,
        ).run(max_iter=50)
        # Our optimiser runs to convergence where the C code stops at 50
        # iterations while still improving, so it should be at least as good.
        assert result.chi2 <= self.REF_CHI2 * 1.02
        assert result.chi2 > 0.5 * self.REF_CHI2

    def test_recovers_the_reference_pole_and_period(self, setup):
        data, t0, law, geometry = setup
        start = SpinState(220.0, 0.0, 5.76198, t0, 0.0)  # the shipped initial guess
        result = HarmonicInversion(
            data, geometry, start, lmax=6, law=law,
            objective=Objective.RELATIVE, convexity_weight=0.1,
            fit_pole=True, fit_period=True,
        ).run(max_iter=60)
        assert result.spin.lam == pytest.approx(self.REF_LAMBDA, abs=2.0)
        assert result.spin.beta == pytest.approx(self.REF_BETA, abs=2.0)
        assert result.spin.period == pytest.approx(self.REF_PERIOD, abs=1e-4)
        assert result.chi2 <= self.REF_CHI2
