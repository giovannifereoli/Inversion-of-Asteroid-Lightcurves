"""Bayesian convex inversion with ``emcee``.

The paper solves the inverse problem by optimisation and controls its
ill-posedness with positivity: Eq. (6) makes every ``g_j`` positive by
construction, and Section 5 concludes that "the positivity constraint is quite
sufficient by itself for removing the apparent ill-posedness of the problem.
No particular regularization methods are necessary".

That argument gives a *point* estimate.  This module keeps the same forward
model - Eqs. (2), (4), (8) and (10) - and the same exponential positivity, but
samples the posterior instead of minimising, which buys three things the paper
notes are otherwise awkward:

* honest uncertainties on the pole and period, rather than the paper's
  suggestion in Section 3.5 to "repeat with different initial values" to gauge
  the spread;
* an explicit noise level, marginalised over instead of assumed;
* the convexity constraint Eq. (3) expressed as a *prior* rather than as extra
  rows in ``chi^2``, so its strength is stated in probabilistic terms.

The natural pole prior is isotropic, ``p(beta) ~ cos beta``, which is what
:class:`BayesianInversion` uses.

Because the sampler must explore every coefficient, the harmonic series is
truncated much earlier here than in :class:`~lcinv.convex.HarmonicInversion`;
``lmax = 3`` (16 coefficients) keeps the dimension near 20 and is usually
enough for the overall dimensions that dominate a lightcurve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .convex import FacetGeometry, HarmonicInversion, Objective
from .geometry import SpinState
from .lightcurve import LightcurveSet
from .minkowski import MinkowskiResult, minkowski_solve
from .scattering import LommelSeeligerLambert, ScatteringLaw

__all__ = ["BayesianInversion", "BayesResult"]


@dataclass
class BayesResult:
    """Posterior samples and summaries.

    Attributes
    ----------
    samples:
        ``(n_samples, n_dim)`` flattened post-burn-in chain.
    labels:
        Parameter names, aligned with the columns of ``samples``.
    log_prob:
        Log-posterior for each sample.
    acceptance_fraction:
        Mean acceptance fraction across walkers.
    autocorr_time:
        Integrated autocorrelation time per parameter, or ``None`` if the
        chain was too short for ``emcee`` to estimate it.
    best:
        The maximum-a-posteriori sample.
    """

    samples: np.ndarray
    labels: list[str]
    log_prob: np.ndarray
    acceptance_fraction: float
    autocorr_time: np.ndarray | None
    best: np.ndarray
    _owner: BayesianInversion | None = field(default=None, repr=False)

    def summary(self, quantiles: tuple[float, float, float] = (16.0, 50.0, 84.0)) -> dict:
        """Median and 1-sigma-equivalent percentile ranges per parameter."""
        pct = np.percentile(self.samples, quantiles, axis=0)
        return {
            name: {
                "median": float(pct[1, i]),
                "lower": float(pct[1, i] - pct[0, i]),
                "upper": float(pct[2, i] - pct[1, i]),
            }
            for i, name in enumerate(self.labels)
        }

    def spin_samples(self) -> np.ndarray:
        """``(n_samples, 3)`` of ``[lambda, beta, period]``, where sampled."""
        if self._owner is None:  # pragma: no cover - defensive
            raise RuntimeError("result is not attached to an inversion")
        return np.column_stack(
            [self.samples[:, i] for i in self._owner._spin_columns()]
        )

    def shape_samples(self, n: int = 20, seed: int | None = 0, **kwargs) -> list[MinkowskiResult]:
        """Reconstruct ``n`` bodies drawn at random from the posterior.

        This is the payoff of sampling: instead of one convex solution, a
        spread of them, whose scatter shows which features the data actually
        constrain.
        """
        if self._owner is None:  # pragma: no cover - defensive
            raise RuntimeError("result is not attached to an inversion")
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(self.samples), size=min(n, len(self.samples)), replace=False)
        return [self._owner.shape_from_sample(self.samples[i], **kwargs) for i in picks]


class BayesianInversion:
    """Posterior sampling over shape, rotation, scattering and noise.

    Parameters
    ----------
    data:
        Observations.
    geometry:
        Normal directions, as for :class:`~lcinv.convex.HarmonicInversion`.
    spin:
        Initial rotation state; also the centre of the pole/period priors.
    lmax:
        Harmonic truncation.  Keep it small - the sampler explores every
        coefficient.
    law:
        Scattering law.
    objective:
        Which residual definition to use; ``RELATIVE`` (Eq. 13) by default.
    fit_pole, fit_period, fit_scattering:
        Which extra blocks to sample.
    period_window:
        Half-width of the uniform period prior, in hours.  The chi-squared
        surface in period is densely multimodal, so this should be narrow and
        centred on a value from a period scan.
    coefficient_scale:
        Standard deviation of the zero-mean Gaussian prior on the harmonic
        coefficients (excluding the constant term).  This is the Bayesian
        counterpart of a smoothness prior: it keeps the curvature function from
        developing arbitrarily fine structure.
    convexity_sigma:
        Prior standard deviation on ``|sum_j n_j g_j| / sum_j g_j``, Eq. (3).
        ``None`` disables it.  Section 3.5 measures this ratio at 0.001-0.007
        for bodies of constant albedo, which is the scale to use.
    """

    def __init__(
        self,
        data: LightcurveSet,
        geometry: FacetGeometry,
        spin: SpinState,
        lmax: int = 3,
        law: ScatteringLaw | None = None,
        objective: Objective | str = Objective.RELATIVE,
        fit_pole: bool = True,
        fit_period: bool = False,
        fit_scattering: bool = False,
        period_window: float = 1e-3,
        coefficient_scale: float = 2.0,
        convexity_sigma: float | None = 0.01,
    ) -> None:
        # The deterministic inversion supplies the forward model, so the two
        # routes cannot drift apart.
        self._fwd = HarmonicInversion(
            data, geometry, spin, lmax=lmax, law=law or LommelSeeligerLambert(0.1),
            objective=objective, convexity_weight=0.0, convexity_components="none",
            fit_pole=fit_pole, fit_period=fit_period, fit_scattering=fit_scattering,
        )
        self.data = data
        self.geometry = geometry
        self.spin = spin
        self.lmax = int(lmax)
        self.fit_pole = bool(fit_pole)
        self.fit_period = bool(fit_period)
        self.fit_scattering = bool(fit_scattering)
        self.period_window = float(period_window)
        self.coefficient_scale = float(coefficient_scale)
        self.convexity_sigma = convexity_sigma
        self.n_data = data.n_points

        self._n_coef = self._fwd.n_coefficients
        self._degrees = np.array(
            [l for l in range(self.lmax + 1) for _ in range(-l, l + 1)]
        )
        # Fix the scale term now so the posterior can be evaluated before run().
        self._fwd._fixed_a00 = float(self._fwd.initial_coefficients(1.3, 1.0, 0.9)[0])

    # ------------------------------------------------------------------
    @property
    def labels(self) -> list[str]:
        """Names of the sampled parameters, in order."""
        names = [f"a[{l},{m}]" for l in range(self.lmax + 1) for m in range(-l, l + 1)]
        if self.fit_pole:
            names += ["lambda", "beta"]
        if self.fit_period:
            names += ["period"]
        if self.fit_scattering:
            names += [f"law{i}" for i in range(len(np.atleast_1d(self._fwd.model.law.parameters)))]
        return names + ["log_sigma"]

    @property
    def n_dim(self) -> int:
        """Number of sampled parameters."""
        return len(self.labels)

    def _spin_columns(self) -> list[int]:
        cols = []
        pos = self._n_coef
        if self.fit_pole:
            cols += [pos, pos + 1]
            pos += 2
        if self.fit_period:
            cols += [pos]
        return cols

    # ------------------------------------------------------------------
    def log_prior(self, theta: np.ndarray) -> float:
        """Log prior density."""
        coeffs = theta[: self._n_coef]
        log_sigma = theta[-1]
        if not -12.0 < log_sigma < 2.0:
            return -np.inf

        # Weakly informative Gaussian on the shape coefficients; the constant
        # term is a scale factor and is left flat.
        varying = coeffs[self._degrees > 0]
        lp = -0.5 * float(varying @ varying) / self.coefficient_scale**2

        pos = self._n_coef
        if self.fit_pole:
            beta = theta[pos + 1]
            pos += 2
            if not -90.0 <= beta <= 90.0:
                return -np.inf
            # Isotropic pole prior: equal probability per unit solid angle.
            # Longitude is uniform, so it contributes only a constant.
            lp += np.log(max(np.cos(np.radians(beta)), 1e-12))
        if self.fit_period:
            period = theta[pos]
            pos += 1
            if abs(period - self.spin.period) > self.period_window:
                return -np.inf
        if self.fit_scattering:
            n_law = len(np.atleast_1d(self._fwd.model.law.parameters))
            law_par = theta[pos : pos + n_law]
            pos += n_law
            if np.any(~np.isfinite(law_par)):
                return -np.inf
        return float(lp)

    def _residuals(self, theta: np.ndarray) -> np.ndarray | None:
        try:
            return self._fwd._residual_fn(theta[:-1])
        except (ValueError, FloatingPointError):  # pragma: no cover - bad proposal
            return None

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Gaussian log likelihood with the noise level as a free parameter."""
        res = self._residuals(theta)
        if res is None or not np.all(np.isfinite(res)):
            return -np.inf
        sigma = float(np.exp(theta[-1]))
        n = len(res)
        ll = -0.5 * float(res @ res) / sigma**2 - n * np.log(sigma) - 0.5 * n * np.log(2.0 * np.pi)

        if self.convexity_sigma is not None:
            coeffs, _, _ = self._fwd._unpack(theta[:-1])
            areas = self._fwd.areas_from_coefficients(coeffs)
            ratio = float(np.linalg.norm(areas @ self.geometry.normals) / max(areas.sum(), 1e-300))
            ll += -0.5 * (ratio / self.convexity_sigma) ** 2
        return ll

    def log_probability(self, theta: np.ndarray) -> float:
        """Log posterior, up to a constant."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    # ------------------------------------------------------------------
    def initial_state(
        self,
        n_walkers: int,
        seed: int | None = 0,
        axes: tuple[float, float, float] = (1.3, 1.0, 0.9),
        start: np.ndarray | None = None,
        pole_scatter: float = 1.0,
    ) -> np.ndarray:
        """A tight ball of walkers around a starting point.

        Parameters
        ----------
        n_walkers:
            Number of walkers.
        seed:
            Random seed.
        axes:
            Semi-axes of the ellipsoid whose curvature function seeds the
            harmonic coefficients, as in Section 3.2.
        start:
            Optional full parameter vector to centre on - pass the optimiser's
            solution to start the chain already converged, which is much more
            efficient than letting it find the mode.
        pole_scatter:
            Initial spread of the pole coordinates, in degrees.
        """
        rng = np.random.default_rng(seed)
        coeffs = self._fwd.initial_coefficients(*axes)
        self._fwd._fixed_a00 = float(coeffs[0])
        centre = [coeffs]
        if self.fit_pole:
            centre.append([self.spin.lam, self.spin.beta])
        if self.fit_period:
            centre.append([self.spin.period])
        if self.fit_scattering:
            centre.append(np.atleast_1d(self._fwd.model.law.parameters))
        centre.append([np.log(0.02)])
        p0 = np.concatenate([np.atleast_1d(np.asarray(c, dtype=float)) for c in centre])
        if start is not None:
            p0 = np.asarray(start, dtype=float).copy()
            if len(p0) != self.n_dim:
                raise ValueError(f"start must have {self.n_dim} entries")

        scatter = np.full(len(p0), 0.02)
        pos = self._n_coef
        if self.fit_pole:
            scatter[pos : pos + 2] = pole_scatter
            pos += 2
        if self.fit_period:
            scatter[pos] = 0.2 * self.period_window
            pos += 1
        scatter[-1] = 0.05
        state = p0 + scatter * rng.standard_normal((n_walkers, len(p0)))
        if self.fit_pole:
            col = self._n_coef + 1
            state[:, col] = np.clip(state[:, col], -89.9, 89.9)
        return state

    def run(
        self,
        n_walkers: int | None = None,
        n_steps: int = 2000,
        burn: int | None = None,
        thin: int = 1,
        seed: int | None = 0,
        progress: bool = False,
        pool=None,
        start: np.ndarray | None = None,
        pole_scatter: float = 1.0,
    ) -> BayesResult:
        """Sample the posterior.

        Parameters
        ----------
        n_walkers:
            Number of walkers; defaults to ``max(2 n_dim + 2, 32)``, the
            minimum ``emcee`` recommends.
        n_steps:
            Steps per walker.
        burn:
            Steps to discard; defaults to half of ``n_steps``.
        thin:
            Keep every ``thin``-th sample.
        seed:
            Seed for the walker initialisation and the sampler.
        progress:
            Show a progress bar.
        pool:
            Optional multiprocessing pool.
        start:
            Full parameter vector to initialise around; see
            :meth:`initial_state`.
        pole_scatter:
            Initial pole spread in degrees.

        Returns
        -------
        BayesResult
        """
        import emcee

        n_dim = self.n_dim
        if n_walkers is None:
            n_walkers = max(2 * n_dim + 2, 32)
        if n_walkers < 2 * n_dim:
            raise ValueError(f"emcee needs at least {2 * n_dim} walkers for {n_dim} parameters")
        if burn is None:
            burn = n_steps // 2

        state = self.initial_state(
            n_walkers, seed=seed, start=start, pole_scatter=pole_scatter
        )
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, self.log_probability, pool=pool,
            # A sharp, strongly correlated posterior: the differential-evolution
            # moves follow its ridges far better than a plain stretch move, but
            # keeping some stretch avoids the ensemble collapsing.
            moves=[
                (emcee.moves.DEMove(), 0.6),
                (emcee.moves.DESnookerMove(), 0.2),
                (emcee.moves.StretchMove(a=1.5), 0.2),
            ],
        )
        sampler.random_state = np.random.default_rng(seed).bit_generator.state
        sampler.run_mcmc(state, n_steps, progress=progress)

        try:
            tau = sampler.get_autocorr_time(quiet=True)
        except Exception:  # pragma: no cover - very short chains
            tau = None
        flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
        logp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
        return BayesResult(
            samples=flat,
            labels=self.labels,
            log_prob=logp,
            acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
            autocorr_time=tau,
            best=flat[int(np.argmax(logp))],
            _owner=self,
        )

    def start_from_result(self, result, sigma: float | None = None) -> np.ndarray:
        """Build a starting vector from a :class:`~lcinv.convex.InversionResult`.

        Passing this to :meth:`run` begins the chain at the optimiser's
        solution, so the burn-in only has to explore the mode rather than find
        it.
        """
        params = np.asarray(result.parameters, dtype=float)
        if len(params) != self.n_dim - 1:
            raise ValueError(
                f"result has {len(params)} parameters, expected {self.n_dim - 1}; "
                "the harmonic degree or free-parameter blocks must match"
            )
        if sigma is None:
            sigma = max(float(result.rms), 1e-6)
        return np.concatenate([params, [np.log(sigma)]])

    def areas_from_sample(self, theta: np.ndarray) -> np.ndarray:
        """Facet values ``g_j`` implied by one posterior sample."""
        coeffs, _, _ = self._fwd._unpack(theta[:-1])
        return self._fwd.areas_from_coefficients(coeffs)

    def spin_from_sample(self, theta: np.ndarray) -> SpinState:
        """The rotation state implied by one posterior sample."""
        _, spin, _ = self._fwd._unpack(theta[:-1])
        return spin

    def shape_from_sample(self, theta: np.ndarray, **kwargs) -> MinkowskiResult:
        """Reconstruct the convex body implied by one posterior sample."""
        return minkowski_solve(self.geometry.normals, self.areas_from_sample(theta), **kwargs)
