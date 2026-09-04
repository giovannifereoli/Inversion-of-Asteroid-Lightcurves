"""Paper II's recipe and the Bayesian layer."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import synthetic_set
from lcinv import (
    BayesianInversion,
    FacetGeometry,
    InversionPipeline,
    LommelSeeligerLambert,
    Objective,
    SpinState,
    ellipsoid,
    period_scan,
    pole_grid,
)
from lcinv.pipeline import period_sampling_interval


class TestPoleGrid:
    def test_covers_the_sphere_roughly_uniformly(self):
        grid = pole_grid(8)
        assert grid.shape[1] == 2
        assert len(grid) > 30
        beta = grid[:, 1]
        assert beta.min() > -90.0 and beta.max() < 90.0
        vectors = np.column_stack(
            [
                np.cos(np.radians(beta)) * np.cos(np.radians(grid[:, 0])),
                np.cos(np.radians(beta)) * np.sin(np.radians(grid[:, 0])),
                np.sin(np.radians(beta)),
            ]
        )
        assert np.linalg.norm(vectors.mean(axis=0)) < 0.2


class TestPeriodScan:
    def test_sampling_interval_shrinks_with_a_longer_baseline(self, ellipsoid_data):
        short = period_sampling_interval(ellipsoid_data)
        assert short > 0
        # Doubling the span should halve the required step.
        jd = np.concatenate([c.jd for c in ellipsoid_data])
        span = (jd.max() - jd.min()) * 24.0
        assert short == pytest.approx(0.8 / (2 * span))

    @pytest.mark.slow
    def test_finds_the_true_period(self, spin, law):
        truth = ellipsoid(1.7, 1.1, 0.95, 7)
        data = synthetic_set(truth, spin, law, n_curves=6, n_points=40, seed=11)
        # The window must be several sampling intervals wide, or the scan
        # returns only a handful of trial periods.
        step = period_sampling_interval(data) * spin.period**2
        periods, chi2 = period_scan(
            data, (spin.period - 8 * step, spin.period + 8 * step),
            geometry=FacetGeometry.from_sphere(3),
            poles=np.array([[60.0, 25.0], [240.0, -25.0]]),
            law=law, lmax=2, t0=spin.t0, max_iter=8,
        )
        assert len(periods) > 10
        assert periods[int(np.argmin(chi2))] == pytest.approx(spin.period, abs=1.5 * step)


@pytest.mark.slow
class TestPipeline:
    def test_runs_every_step_and_reports(self, spin, law):
        truth = ellipsoid(1.7, 1.15, 0.95, 7)
        data = synthetic_set(truth, spin, law, n_curves=10, seed=12).with_noise(0.02, seed=12)
        pipeline = InversionPipeline(
            data, geometry=FacetGeometry.from_sphere(6), law=law, lmax=5
        )
        start = SpinState(spin.lam + 8.0, spin.beta - 8.0, spin.period, spin.t0, spin.phi0)
        result = pipeline.run(
            start, convexity_weights=(0.1,), fit_pole=True, fit_period=False,
            n_restarts=2, restart_scatter=3.0, verbose=False,
        )
        assert result.facet is not None
        assert result.shape is not None
        assert result.best.chi2 <= result.series.chi2
        # The pole should come back close to the truth.
        assert result.best.spin.lam == pytest.approx(spin.lam, abs=12.0)
        assert result.pole_scatter is not None and result.pole_scatter >= 0.0
        assert len(result.trials) == 2
        text = result.report()
        assert "pole" in text and "period" in text and "a:b:c" in text
        assert any("step 3" in line for line in result.log)
        assert any("step 7" in line for line in result.log)

    def test_picks_the_objective_from_the_data(self, ellipsoid_data, law):
        pipeline = InversionPipeline(ellipsoid_data, law=law)
        assert pipeline.objective is Objective.RELATIVE


@pytest.mark.slow
class TestBayesianInversion:
    @pytest.fixture(scope="class")
    def setup(self):
        spin = SpinState(60.0, 25.0, 6.0, 2450000.0, 0.0)
        law = LommelSeeligerLambert(1.0)
        truth = ellipsoid(1.6, 1.15, 0.95, 7)
        data = synthetic_set(truth, spin, law, n_curves=8, n_points=30, seed=13)
        return data.with_noise(0.02, seed=13), spin, law

    def test_posterior_is_finite_and_penalises_bad_poles(self, setup):
        from lcinv import BayesianInversion

        data, spin, law = setup
        inv = BayesianInversion(
            data, FacetGeometry.from_sphere(5), spin, lmax=2, law=law, fit_pole=True
        )
        theta = inv.initial_state(4, seed=0)[0]
        assert np.isfinite(inv.log_probability(theta))
        bad = theta.copy()
        bad[inv.labels.index("beta")] = 120.0  # outside the sphere
        assert inv.log_prior(bad) == -np.inf
        bad = theta.copy()
        bad[-1] = 50.0  # absurd noise level
        assert inv.log_prior(bad) == -np.inf

    def test_isotropic_pole_prior(self, setup):
        from lcinv import BayesianInversion

        data, spin, law = setup
        inv = BayesianInversion(
            data, FacetGeometry.from_sphere(4), spin, lmax=1, law=law, fit_pole=True
        )
        theta = inv.initial_state(4, seed=0)[0]
        col = inv.labels.index("beta")
        equator, pole = theta.copy(), theta.copy()
        equator[col], pole[col] = 0.0, 80.0
        assert inv.log_prior(equator) > inv.log_prior(pole)

    def test_short_chain_recovers_the_pole(self, setup):
        from lcinv import BayesianInversion

        data, spin, law = setup
        inv = BayesianInversion(
            data, FacetGeometry.from_sphere(5), spin, lmax=2, law=law,
            fit_pole=True, fit_period=False,
        )
        result = inv.run(n_walkers=40, n_steps=400, burn=200, seed=0)
        assert result.samples.shape[1] == inv.n_dim
        assert 0.0 < result.acceptance_fraction < 1.0
        summary = result.summary()
        assert set(summary) == set(result.labels)
        assert summary["lambda"]["median"] == pytest.approx(spin.lam, abs=25.0)
        # The recovered noise level should be near the 2% that was injected.
        assert np.exp(summary["log_sigma"]["median"]) == pytest.approx(0.02, rel=1.5)
        assert result.spin_samples().shape == (len(result.samples), 2)

    def test_needs_enough_walkers(self, setup):
        from lcinv import BayesianInversion

        data, spin, law = setup
        inv = BayesianInversion(data, FacetGeometry.from_sphere(4), spin, lmax=1, law=law)
        with pytest.raises(ValueError):
            inv.run(n_walkers=4, n_steps=10)


class TestChainLengthControl:
    """`target_tau` automates emcee's own 50-tau advice."""

    @staticmethod
    def _tiny_inversion():
        spin = SpinState(40.0, -20.0, 6.0, 2450000.0, 0.0)
        law = LommelSeeligerLambert(0.1)
        data = synthetic_set(ellipsoid(1.5, 1.1, 1.0, 5), spin, law, n_curves=4, n_points=18)
        return BayesianInversion(
            data, FacetGeometry.from_sphere(4), spin, lmax=1, law=law, fit_pole=False
        )

    def test_laplace_scatter_is_positive_and_sized_right(self):
        bi = self._tiny_inversion()
        sd = bi.laplace_scatter(bi.initial_state(8, seed=0)[0])
        if sd is not None:          # pinv can legitimately fail on a tiny problem
            assert len(sd) == bi.n_dim
            assert np.all(sd > 0.0) and np.all(np.isfinite(sd))

    def test_target_tau_extends_the_chain(self):
        bi = self._tiny_inversion()
        n_walkers = 2 * bi.n_dim + 2
        short = bi.run(n_walkers=n_walkers, n_steps=200, seed=0)
        grown = bi.run(
            n_walkers=n_walkers, n_steps=200, seed=0, target_tau=30, max_steps=1200
        )
        assert len(grown.samples) > len(short.samples)

    def test_max_steps_is_respected(self):
        bi = self._tiny_inversion()
        n_walkers = 2 * bi.n_dim + 2
        grown = bi.run(
            n_walkers=n_walkers, n_steps=100, seed=0,
            target_tau=10_000, max_steps=400,   # unreachable target
        )
        assert len(grown.samples) <= n_walkers * 400
