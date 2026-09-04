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

        Parameters
        ----------
        n:
            Number of posterior draws to reconstruct.
        seed:
            Seed for choosing which samples to draw.
        **kwargs:
            Passed to :func:`~lcinv.minkowski.minkowski_solve`.

        Returns
        -------
        list of ~lcinv.minkowski.MinkowskiResult
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

        degrees = np.array([l for l in range(self.lmax + 1) for _ in range(-l, l + 1)])
        # Under Eq. (13) the constant term is a pure scale factor, so the
        # forward model overwrites it with a fixed value (Section 3.4).  An
        # optimiser can carry such a column harmlessly - its Jacobian entry is
        # zero and it never moves - but a sampler cannot: with no likelihood and
        # no prior the direction is improper, the ensemble diffuses along it
        # without bound, and because the differential-evolution moves scale
        # their proposals by the ensemble's own spread, that runaway degrades
        # the mixing of every other parameter.  So it is left out of the
        # sampled vector entirely and reinserted before each forward call.
        self._skip_a00 = bool(self._fwd.fix_scale)
        self._degrees = degrees[1:] if self._skip_a00 else degrees
        self._n_coef = len(self._degrees)
        # Fix the scale term now so the posterior can be evaluated before run().
        self._fwd._fixed_a00 = float(self._fwd.initial_coefficients(1.3, 1.0, 0.9)[0])

    def _forward_params(self, theta: np.ndarray) -> np.ndarray:
        """The forward model's parameter vector for a sample ``theta``.

        ``theta`` carries no ``log_sigma`` tail and, when the scale term is
        fixed, no ``a[0,0]``; :class:`~lcinv.convex.HarmonicInversion` expects
        both slots present, so put the placeholder back.
        """
        params = np.asarray(theta, dtype=float)[:-1]
        if self._skip_a00:
            params = np.concatenate([[self._fwd._fixed_a00], params])
        return params

    # ------------------------------------------------------------------
    @property
    def labels(self) -> list[str]:
        """Names of the sampled parameters, in order."""
        names = [f"a[{l},{m}]" for l in range(self.lmax + 1) for m in range(-l, l + 1)]
        if self._skip_a00:
            names = names[1:]
        if self.fit_pole:
            names += ["lambda", "beta"]
        if self.fit_period:
            names += ["period"]
        if self.fit_scattering:
            names += list(self._law_names)
        return names + ["log_sigma"]

    @property
    def _law_mask(self) -> np.ndarray:
        """Which scattering parameters the forward model actually varies.

        :class:`~lcinv.convex.HarmonicInversion` holds the unidentifiable ones
        fixed (Eq. 13 cancels any pure scale factor), so the sampler must use
        the same mask or its parameter vector will not line up with the one the
        forward model unpacks.
        """
        return np.asarray(self._fwd.model.law.free_parameter_mask, dtype=bool)

    @property
    def _law_names(self) -> list[str]:
        """Names of the sampled scattering parameters, in order."""
        return [f"law{i}" for i in np.flatnonzero(self._law_mask)]

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

        # Weakly informative Gaussian on the shape coefficients.  A constant
        # term still in the vector (absolute photometry) is a scale factor and
        # is left flat; when the objective is relative it is not sampled at all.
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
            mask = self._law_mask
            n_law = int(mask.sum())
            law_par = theta[pos : pos + n_law]
            pos += n_law
            if np.any(~np.isfinite(law_par)):
                return -np.inf
            # Uniform inside the law's own physical limits, zero outside.  The
            # optimiser gets these as box constraints; the sampler has to get
            # them as a prior or its walkers will wander into negative surge
            # widths and unphysical slopes that happen to fit slightly better.
            lo, hi = self._fwd.model.law.parameter_bounds
            if np.any(law_par < np.asarray(lo, float)[mask]) or np.any(
                law_par > np.asarray(hi, float)[mask]
            ):
                return -np.inf
        return float(lp)

    def _residuals(self, theta: np.ndarray) -> np.ndarray | None:
        try:
            return self._fwd._residual_fn(self._forward_params(theta))
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
            coeffs, _, _ = self._fwd._unpack(self._forward_params(theta))
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
        laplace: bool = False,
    ) -> np.ndarray:
        """A ball of walkers around a starting point, scaled to the posterior.

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
            Initial spread of the pole coordinates, in degrees.  Ignored when
            ``laplace`` is in use.
        laplace:
            Scale each parameter's initial spread by the Laplace (Gaussian)
            approximation to its posterior width, from
            :meth:`laplace_scatter`.  An ensemble started at the wrong scale
            has to expand or contract before it can mix, which can show up as a
            long integrated autocorrelation time.

            Off by default: on the (269) Justitia problem it made no
            measurable difference, because the fixed scatters already matched
            the posterior widths to within a factor of two.  It is worth trying
            when a chain mixes badly and the parameter scales are unknown.
        """
        rng = np.random.default_rng(seed)
        coeffs = self._fwd.initial_coefficients(*axes)
        self._fwd._fixed_a00 = float(coeffs[0])
        centre = [coeffs[1:] if self._skip_a00 else coeffs]
        if self.fit_pole:
            centre.append([self.spin.lam, self.spin.beta])
        if self.fit_period:
            centre.append([self.spin.period])
        if self.fit_scattering:
            centre.append(np.atleast_1d(self._fwd.model.law.parameters)[self._law_mask])
        centre.append([np.log(0.02)])
        p0 = np.concatenate([np.atleast_1d(np.asarray(c, dtype=float)) for c in centre])
        if start is not None:
            p0 = np.asarray(start, dtype=float).copy()
            if len(p0) != self.n_dim:
                raise ValueError(f"start must have {self.n_dim} entries")

        scatter = self.laplace_scatter(p0) if laplace else None
        if scatter is None:
            scatter = np.full(len(p0), 0.02)
            pos = self._n_coef
            if self.fit_pole:
                scatter[pos : pos + 2] = pole_scatter
                pos += 2
            if self.fit_period:
                scatter[pos] = 0.2 * self.period_window
                pos += 1
            if self.fit_scattering:
                # A single absolute spread cannot suit c, a, d and k at once -
                # d is O(0.1) with a hard floor just below it - so scale each by
                # its own magnitude.
                law0 = p0[pos : pos + int(self._law_mask.sum())]
                scatter[pos : pos + len(law0)] = np.maximum(0.05 * np.abs(law0), 1e-3)
                pos += len(law0)
            scatter[-1] = 0.05
        state = p0 + scatter * rng.standard_normal((n_walkers, len(p0)))
        if self.fit_pole:
            col = self._n_coef + 1
            state[:, col] = np.clip(state[:, col], -89.9, 89.9)
        if self.fit_scattering:
            # A walker started outside the law's bounds has zero prior and can
            # never move, so the ensemble would start a member short.
            col = self._n_coef + 2 * self.fit_pole + self.fit_period
            n_law = int(self._law_mask.sum())
            lo, hi = self._fwd.model.law.parameter_bounds
            lo, hi = np.asarray(lo, float)[self._law_mask], np.asarray(hi, float)[self._law_mask]
            span = np.where(np.isfinite(hi - lo), 1e-6 * (hi - lo), 1e-6)
            state[:, col : col + n_law] = np.clip(
                state[:, col : col + n_law], lo + span, hi - span
            )
        return state

    def laplace_scatter(self, theta: np.ndarray, fraction: float = 0.5) -> np.ndarray | None:
        """Per-parameter posterior widths from the Gaussian (Laplace) approximation.

        At the optimum the covariance is ``sigma^2 (J^T J)^-1`` with ``J`` the
        Jacobian of the residuals, which the deterministic inversion already
        computes analytically.  Returns ``fraction`` of those standard
        deviations, so the walkers start comfortably inside the posterior
        rather than having to find its scale by random walk.

        Parameters
        ----------
        theta:
            Full parameter vector to linearise about, normally the optimiser's
            solution.
        fraction:
            Fraction of each standard deviation to return.

        Returns
        -------
        numpy.ndarray or None
            ``(n_dim,)`` widths, or ``None`` when the Jacobian is unusable so
            that callers can fall back to fixed scatters.
        """
        try:
            params = self._forward_params(theta)
            jac = self._fwd._jacobian_fn(params)
            res = self._fwd._residual_fn(params)
            if self._skip_a00:
                # That column is identically zero, so it would make J^T J
                # singular in a way the pseudo-inverse reports as an infinite
                # width for a parameter that is not even sampled.
                jac = jac[:, 1:]
                params = params[1:]
            dof = max(len(res) - len(params), 1)
            sigma2 = float(res @ res) / dof
            cov = sigma2 * np.linalg.pinv(jac.T @ jac, rcond=1e-12)
            sd = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
            if not np.all(np.isfinite(sd)) or np.all(sd == 0.0):
                return None
            # log_sigma has an analytic width for a Gaussian likelihood.
            sd = np.concatenate([sd, [1.0 / np.sqrt(2.0 * len(res))]])
            floor = 1e-8 * max(float(np.max(np.abs(theta))), 1.0)
            return fraction * np.maximum(sd, floor)
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover
            return None

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
        laplace: bool = False,
        target_tau: float = 0.0,
        max_steps: int | None = None,
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
            n_walkers, seed=seed, start=start, pole_scatter=pole_scatter, laplace=laplace
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

        if target_tau > 0.0:
            ceiling = max_steps if max_steps is not None else 20 * n_steps
            while sampler.iteration < ceiling:
                try:
                    tau_now = float(np.max(sampler.get_autocorr_time(quiet=True)))
                except Exception:  # pragma: no cover - very short chains
                    break
                if not np.isfinite(tau_now) or sampler.iteration >= target_tau * tau_now:
                    break
                extra = int(
                    min(
                        ceiling - sampler.iteration,
                        max(n_steps // 2, target_tau * tau_now - sampler.iteration),
                    )
                )
                if extra <= 0:
                    break
                sampler.run_mcmc(None, extra, progress=progress)
            burn = sampler.iteration // 2

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

        Parameters
        ----------
        result:
            A converged :class:`~lcinv.convex.InversionResult` whose free
            parameters match this inversion's.
        sigma:
            Starting noise level; taken from the result's residual scatter
            when omitted.

        Returns
        -------
        numpy.ndarray
            ``(n_dim,)`` starting vector, including ``log_sigma``.
        """
        params = np.asarray(result.parameters, dtype=float)
        if self._skip_a00 and len(params) == self.n_dim:
            params = params[1:]
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
        coeffs, _, _ = self._fwd._unpack(self._forward_params(theta))
        return self._fwd.areas_from_coefficients(coeffs)

    def spin_from_sample(self, theta: np.ndarray) -> SpinState:
        """The rotation state implied by one posterior sample."""
        _, spin, _ = self._fwd._unpack(self._forward_params(theta))
        return spin

    def shape_from_sample(self, theta: np.ndarray, **kwargs) -> MinkowskiResult:
        """Reconstruct the convex body implied by one posterior sample.

        Parameters
        ----------
        theta:
            One row of :attr:`BayesResult.samples`.
        **kwargs:
            Passed to :func:`~lcinv.minkowski.minkowski_solve`.

        Returns
        -------
        ~lcinv.minkowski.MinkowskiResult
        """
        return minkowski_solve(self.geometry.normals, self.areas_from_sample(theta), **kwargs)
