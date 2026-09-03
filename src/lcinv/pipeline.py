"""The complete inversion recipe.

Paper I determines the shape with the rotation state given.  Its companion,

    M. Kaasalainen, J. Torppa and K. Muinonen (2001), "Optimization Methods
    for Asteroid Lightcurve Inversion. II. The Complete Inverse Problem",
    Icarus 153, 37-51,

closes the loop and states the procedure as a numbered recipe, which this
module implements:

    1. Determine the sampling interval of the period from the separation
       between the local chi^2(P) minima [...] or by plotting chi^2(P) using,
       e.g., an ellipsoidal model.
    2. [Choose the initial pole] from a suitable grid and/or a previous model.
       Choose the scattering law and its initial parameters.
    3. Use the function series method of Paper I, adding pole, period, and
       scattering parameters to the Levenberg-Marquardt procedure.  Try
       different convexity (and/or scale factor) regularization weights if
       necessary.  Pick the best solution of the set.
    4. Refine the shape result by switching to the separate facet method of
       Paper I, using the facet values from step 3 as the starting point.
    5. If the best solution contains real nonnegligible residual nonconvexity,
       separate the shape solution and albedo asymmetry using convexity
       constraint (and smoothness regularization) as shown in Paper I.
    6. Plot the scale factors to determine the solar phase function f(alpha).
    7. Find the shape from the facet areas by Minkowski minimization (Paper I).
    8. Repeat steps 2-7 with different (but nearby) initial values and
       scattering laws to obtain error estimates.

Steps 3, 4, 5 and 7 are Paper I's own machinery
(:class:`~lcinv.convex.HarmonicInversion`,
:class:`~lcinv.convex.FacetInversion`,
:class:`~lcinv.albedo.AlbedoSeparation`,
:func:`~lcinv.minkowski.minkowski_solve`); this module supplies the scan, the
pole grid and the bookkeeping that joins them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .albedo import AlbedoResult, AlbedoSeparation
from .convex import (
    FacetGeometry,
    FacetInversion,
    HarmonicInversion,
    InversionResult,
    Objective,
)
from .geometry import SpinState
from .lightcurve import LightcurveSet
from .minkowski import MinkowskiResult, minkowski_solve
from .scattering import LommelSeeligerLambert, ScatteringLaw

__all__ = [
    "period_sampling_interval",
    "period_scan",
    "pole_grid",
    "PipelineResult",
    "InversionPipeline",
]


def period_sampling_interval(data: LightcurveSet, coefficient: float = 0.8) -> float:
    """Step 1 - the spacing of the local ``chi^2(P)`` minima, in hours.

    Paper II derives the separation of adjacent period minima from the total
    time span ``T`` of the data: two trial periods become distinguishable once
    the accumulated rotational phase drifts by about half a turn, giving
    ``dP ~ coefficient P^2 / (2 T)``.  Scanning on a grid coarser than this
    steps over minima.

    Parameters
    ----------
    data:
        The observations; only their time span is used.
    coefficient:
        Safety factor; below one oversamples.

    Returns
    -------
    float
        Sampling interval *per unit ``P^2``*, in hours per hour-squared, so
        that the interval at period ``P`` is ``interval * P**2``.
    """
    jd = np.concatenate([c.jd for c in data])
    span_hours = float(jd.max() - jd.min()) * 24.0
    if span_hours <= 0:
        raise ValueError("lightcurves span no time")
    return float(coefficient / (2.0 * span_hours))


def period_scan(
    data: LightcurveSet,
    period_range: tuple[float, float],
    geometry: FacetGeometry | None = None,
    poles: "np.ndarray | None" = None,
    law: ScatteringLaw | None = None,
    lmax: int = 2,
    t0: float | None = None,
    coefficient: float = 0.8,
    max_iter: int = 12,
    objective: Objective | str = Objective.RELATIVE,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Step 1 - ``chi^2`` against trial period.

    Paper II suggests plotting this "using, e.g., an ellipsoidal model (neither
    the ratios of the axes nor the pole need to be accurate for this purpose)",
    so the default is a very low harmonic order and a short optimisation at
    each trial period, taking the best over a coarse pole grid.

    Parameters
    ----------
    data:
        The observations.
    period_range:
        ``(min, max)`` trial period in hours.
    geometry:
        Normal directions; a coarse sphere by default, since resolution is not
        needed here.
    poles:
        ``(K, 2)`` of ``(lambda, beta)`` trial poles; a small grid by default.
    law:
        Scattering law.
    lmax:
        Harmonic truncation for the scan.
    t0:
        Reference epoch; the first observation by default.
    coefficient:
        Passed to :func:`period_sampling_interval`.
    max_iter:
        Levenberg-Marquardt iterations per trial.
    objective:
        Which chi-squared to use.
    progress:
        Print progress every 25 trials.

    Returns
    -------
    periods, chi2:
        The scanned grid and the best chi-squared found at each period.
    """
    geometry = geometry or FacetGeometry.from_sphere(4)
    law = law or LommelSeeligerLambert(0.1)
    poles = np.asarray(
        poles if poles is not None else pole_grid(6), dtype=float
    ).reshape(-1, 2)
    if t0 is None:
        t0 = float(min(c.jd.min() for c in data))

    lo, hi = float(period_range[0]), float(period_range[1])
    if not 0 < lo < hi:
        raise ValueError("period_range must be positive and increasing")
    rate = period_sampling_interval(data, coefficient)

    periods: list[float] = [lo]
    while periods[-1] < hi:
        periods.append(periods[-1] + rate * periods[-1] ** 2)
    grid = np.asarray(periods[:-1] if periods[-1] > hi else periods)

    chi2 = np.empty(len(grid))
    for i, period in enumerate(grid):
        best = np.inf
        for lam, beta in poles:
            inv = HarmonicInversion(
                data, geometry, SpinState(lam, beta, period, t0, 0.0),
                lmax=lmax, law=law, objective=objective,
                convexity_weight=0.1,
            )
            try:
                best = min(best, inv.run(max_iter=max_iter).chi2)
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                continue
        chi2[i] = best
        if progress and i % 25 == 0:  # pragma: no cover - console noise
            print(f"  period scan {i + 1}/{len(grid)}  P={period:.5f} chi2={best:.5f}")
    return grid, chi2


def pole_grid(n: int = 6) -> np.ndarray:
    """Step 2 - a roughly uniform grid of trial pole directions.

    Parameters
    ----------
    n:
        Controls the density; the grid holds about ``2 n^2 / pi`` directions
        spread evenly over the sphere.

    Returns
    -------
    numpy.ndarray
        ``(K, 2)`` of ``(lambda, beta)`` in degrees.
    """
    out: list[tuple[float, float]] = []
    for i in range(n):
        beta = -90.0 + 180.0 * (i + 0.5) / n
        count = max(1, int(round(2 * n * np.cos(np.radians(beta)))))
        for k in range(count):
            out.append((360.0 * k / count, beta))
    return np.asarray(out)


@dataclass
class PipelineResult:
    """Everything the recipe produced.

    Attributes
    ----------
    series:
        Step 3 - the best :class:`~lcinv.convex.InversionResult` from the
        function-series method over the regularisation weights tried.
    facet:
        Step 4 - the facet-method refinement.
    best:
        Whichever of the two fits better; the one to quote.
    albedo:
        Step 5 - the albedo separation, when it was triggered.
    shape:
        Step 7 - the Minkowski reconstruction.
    trials:
        Step 8 - results from the perturbed restarts.
    pole_scatter, period_scatter:
        Step 8 - spread of pole direction (degrees) and period (hours) across
        the restarts, as an empirical error estimate.
    period_scan:
        Step 1 - ``(periods, chi2)`` if a scan was run.
    log:
        Human-readable trace of what each step did.
    """

    series: InversionResult
    facet: InversionResult | None = None
    albedo: AlbedoResult | None = None
    shape: MinkowskiResult | None = None
    trials: list[InversionResult] = field(default_factory=list)
    pole_scatter: float | None = None
    period_scatter: float | None = None
    period_scan: "tuple[np.ndarray, np.ndarray] | None" = None
    log: list[str] = field(default_factory=list)

    @property
    def best(self) -> InversionResult:
        """The better of the series and facet solutions."""
        if self.facet is not None and self.facet.chi2 < self.series.chi2:
            return self.facet
        return self.series

    def report(self) -> str:
        """A printable summary."""
        b = self.best
        lines = [
            f"pole      lambda = {b.spin.lam:8.3f} deg   beta = {b.spin.beta:7.3f} deg",
            f"period    P      = {b.spin.period:.7f} h",
            f"fit       chi2   = {b.chi2:.6f}   rms = {b.rms:.6f}",
            f"residual nonconvexity = {b.nonconvexity:.4f}",
        ]
        if self.pole_scatter is not None:
            lines.append(
                f"restart scatter: pole {self.pole_scatter:.2f} deg, "
                f"period {self.period_scatter:.2e} h"
            )
        if self.albedo is not None:
            lines.append(
                f"albedo asymmetry: {self.albedo.albedo.min():.3f} - "
                f"{self.albedo.albedo.max():.3f}"
            )
        if self.shape is not None:
            ex = self.shape.polyhedron.extents()
            lines.append(
                f"shape     a:b:c = {ex[0] / ex[2]:.3f} : {ex[1] / ex[2]:.3f} : 1.000"
            )
        return "\n".join(lines)


class InversionPipeline:
    """Run the Paper II recipe end to end.

    Parameters
    ----------
    data:
        The observations.
    geometry:
        Normal directions; ``FacetGeometry.from_sphere(8)`` (512 facets) by
        default.  Section 3.5 recommends "of order 1000" for a final model,
        i.e. ``from_sphere(11)``.
    law:
        Initial scattering law.
    objective:
        Which chi-squared; ``RELATIVE`` (Eq. 13) unless the data are calibrated.
    lmax:
        Harmonic truncation for step 3.
    """

    def __init__(
        self,
        data: LightcurveSet,
        geometry: FacetGeometry | None = None,
        law: ScatteringLaw | None = None,
        objective: Objective | str | None = None,
        lmax: int = 6,
    ) -> None:
        self.data = data
        self.geometry = geometry or FacetGeometry.from_sphere(8)
        self.law = law or LommelSeeligerLambert(0.1)
        if objective is None:
            objective = Objective.RENORMALISED if data.all_calibrated else Objective.RELATIVE
        self.objective = Objective(objective)
        self.lmax = int(lmax)
        self.t0 = float(min(c.jd.min() for c in data))

    def run(
        self,
        spin: SpinState,
        convexity_weights: "tuple[float, ...]" = (0.1, 1.0),
        fit_pole: bool = True,
        fit_period: bool = True,
        fit_scattering: bool = False,
        refine_facets: bool = True,
        separate_albedo: "bool | None" = None,
        nonconvexity_threshold: float = 0.01,
        reconstruct: bool = True,
        n_restarts: int = 0,
        restart_scatter: float = 5.0,
        max_iter: int = 60,
        seed: int | None = 0,
        verbose: bool = True,
    ) -> PipelineResult:
        """Execute steps 3-8.

        Parameters
        ----------
        spin:
            Initial rotation state (step 2).  Use :func:`period_scan` and
            :func:`pole_grid` first if it is not already known.
        convexity_weights:
            Step 3 - "try different convexity [...] regularization weights if
            necessary.  Pick the best solution of the set."
        fit_pole, fit_period, fit_scattering:
            Which parameters to add to the Levenberg-Marquardt procedure.
        refine_facets:
            Step 4.
        separate_albedo:
            Step 5; ``None`` triggers it only when the residual nonconvexity
            exceeds ``nonconvexity_threshold``.
        nonconvexity_threshold:
            Section 3.5 measures 0.001-0.007 for constant-albedo bodies, so
            anything above about 0.01 is "real nonnegligible residual
            nonconvexity".
        reconstruct:
            Step 7 - Minkowski minimisation.
        n_restarts:
            Step 8 - how many perturbed restarts to run for error estimates.
        restart_scatter:
            Pole perturbation for those restarts, in degrees.
        max_iter:
            Optimiser iteration cap.
        seed:
            Seed for the restart perturbations.
        verbose:
            Print progress.

        Returns
        -------
        PipelineResult
        """
        log: list[str] = []

        def say(message: str) -> None:
            log.append(message)
            if verbose:
                print(message)

        # --- step 3: function series, with pole/period/scattering free ------
        best: InversionResult | None = None
        for weight in convexity_weights:
            inv = HarmonicInversion(
                self.data, self.geometry, spin, lmax=self.lmax, law=self.law,
                objective=self.objective, convexity_weight=weight,
                fit_pole=fit_pole, fit_period=fit_period, fit_scattering=fit_scattering,
            )
            trial = inv.run(max_iter=max_iter)
            say(
                f"step 3  conv.reg={weight:<5g} chi2={trial.chi2:.6f} rms={trial.rms:.6f} "
                f"lambda={trial.spin.lam:.2f} beta={trial.spin.beta:.2f} P={trial.spin.period:.6f}"
            )
            if best is None or trial.chi2 < best.chi2:
                best = trial
        assert best is not None
        say(f"step 3  best chi2={best.chi2:.6f}, residual nonconvexity={best.nonconvexity:.4f}")

        # --- step 4: facet refinement --------------------------------------
        facet = None
        if refine_facets:
            polish = FacetInversion(
                self.data, self.geometry, best.spin, law=best.law,
                objective=self.objective, convexity_weight=convexity_weights[0],
            )
            facet = polish.run(initial=best.areas, max_iter=800)
            say(
                f"step 4  facet refinement chi2={facet.chi2:.6f} rms={facet.rms:.6f} "
                f"nonconvexity={facet.nonconvexity:.4f}"
            )

        chosen = facet if (facet is not None and facet.chi2 < best.chi2) else best

        # --- step 5: albedo separation -------------------------------------
        albedo = None
        trigger = (
            chosen.nonconvexity > nonconvexity_threshold
            if separate_albedo is None
            else separate_albedo
        )
        if trigger:
            unconstrained = FacetInversion(
                self.data, self.geometry, chosen.spin, law=chosen.law,
                objective=self.objective, convexity_components="none",
            ).run(initial=chosen.areas, max_iter=600)
            albedo = AlbedoSeparation(self.geometry, unconstrained.areas).run(max_nfev=400)
            say(
                f"step 5  albedo separation: varpi in "
                f"[{albedo.albedo.min():.3f}, {albedo.albedo.max():.3f}], "
                f"area-part nonconvexity={albedo.residual_nonconvexity:.2e}"
            )
        else:
            say(
                f"step 5  skipped: residual nonconvexity {chosen.nonconvexity:.4f} "
                f"<= {nonconvexity_threshold}"
            )

        # --- step 7: Minkowski ---------------------------------------------
        shape = None
        if reconstruct:
            shape = minkowski_solve(self.geometry.normals, chosen.areas)
            ex = shape.polyhedron.extents()
            say(
                f"step 7  Minkowski: {shape.n_iterations} iterations, "
                f"alignment={shape.alignment:.6f}, a:b:c = "
                f"{ex[0] / ex[2]:.3f} : {ex[1] / ex[2]:.3f} : 1.000"
            )

        # --- step 8: restarts for error estimates ---------------------------
        trials: list[InversionResult] = []
        pole_scatter = period_scatter = None
        if n_restarts > 0:
            rng = np.random.default_rng(seed)
            for k in range(n_restarts):
                jitter = SpinState(
                    chosen.spin.lam + rng.normal(scale=restart_scatter),
                    float(np.clip(chosen.spin.beta + rng.normal(scale=restart_scatter), -89.9, 89.9)),
                    chosen.spin.period,
                    chosen.spin.t0, chosen.spin.phi0, chosen.spin.yorp,
                )
                inv = HarmonicInversion(
                    self.data, self.geometry, jitter, lmax=self.lmax, law=self.law,
                    objective=self.objective, convexity_weight=convexity_weights[0],
                    fit_pole=fit_pole, fit_period=fit_period,
                )
                trials.append(inv.run(max_iter=max_iter))
                say(f"step 8  restart {k + 1}/{n_restarts} chi2={trials[-1].chi2:.6f}")
            poles = np.array([t.spin.pole_vector() for t in trials] + [chosen.spin.pole_vector()])
            mean = poles.mean(axis=0)
            mean /= max(np.linalg.norm(mean), 1e-300)
            pole_scatter = float(
                np.degrees(np.arccos(np.clip(poles @ mean, -1.0, 1.0))).std()
            )
            period_scatter = float(
                np.std([t.spin.period for t in trials] + [chosen.spin.period])
            )
            say(
                f"step 8  scatter: pole {pole_scatter:.2f} deg, period {period_scatter:.2e} h"
            )

        return PipelineResult(
            series=best, facet=facet, albedo=albedo, shape=shape, trials=trials,
            pole_scatter=pole_scatter, period_scatter=period_scatter, log=log,
        )
