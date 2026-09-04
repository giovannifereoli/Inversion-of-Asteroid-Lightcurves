"""Section 3 - convex inversion.

The convex inverse problem is Eq. (2), ``L = A g``, with Eq. (4)

.. math::  A_{ij} = S_j(\\mu^{(ij)}, \\mu_0^{(ij)})\\, \\varpi_j .

Section 3 opens by noting that solving this linear system directly "would
usually produce negative ``g_j`` values", and that the cure is not
regularisation but positivity:

    The easiest way to guarantee positivity is to represent each ``g_j``
    exponentially, the optimization parameter being now the exponent ``a_j``
    [Eq. (6)].  The values of ``a_j`` are not constrained, so this is much more
    practicable than using penalty or barrier functions.

Two parametrisations follow, and they are "rather complementary":

* :class:`FacetInversion` - Section 3.1, ``g_j = exp(a_j)`` per facet, "of
  order 1000" parameters, minimised by conjugate gradients;
* :class:`HarmonicInversion` - Section 3.2, ``G = exp(sum a_lm Y_lm)`` with
  "typically from, say, 40 to 100" coefficients, minimised by
  Levenberg-Marquardt.

Both minimise one of the three objective functions of the paper, selected by
:class:`Objective`, and both can carry the convexity constraint Eq. (3) as
extra rows, which Section 3.3 describes as "adding three zero elements to L
and three new rows to A".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree

from .geometry import SpinState, unit_to_spherical
from .lightcurve import LightcurveSet
from .mesh import Polyhedron
from .minkowski import MinkowskiResult, minkowski_solve
from .scattering import LommelSeeligerLambert, ScatteringLaw
from .sphharm import design_matrix, n_coefficients
from .triangulation import facet_adjacency, octant_triangulation

__all__ = [
    "Objective",
    "FacetGeometry",
    "ConvexModel",
    "InversionResult",
    "FacetInversion",
    "HarmonicInversion",
    "nonconvexity_residual",
    "ellipsoid_log_curvature",
]


class Objective(Enum):
    """Which chi-squared the paper defines is being minimised."""

    #: Eq. (5), ``chi^2 = |L - A g|^2``.
    ABSOLUTE = "absolute"
    #: Eq. (7), each lightcurve divided by its own mean brightness.
    RENORMALISED = "renormalised"
    #: Eq. (13), *both* observed and model lightcurves scaled to mean unity.
    RELATIVE = "relative"


@dataclass
class FacetGeometry:
    """The fixed set of surface normal directions and their sphere areas.

    Section 3.1: "the unit vector ``n_j`` is the chosen surface outward normal
    of the facet ``j`` (we typically use the facet normals of a sphere or a
    triaxial ellipsoid triangulated in the standard manner)".  Section 3.5 adds
    that "the number of parameters should be of order 1000 (corresponding to
    evenly distributed surface normals) to make the result independent of the
    exact choice of the normal directions".

    Attributes
    ----------
    normals:
        ``(M, 3)`` unit normals ``n_j``.
    sphere_areas:
        ``(M,)`` areas ``Delta sigma_j`` of the corresponding sphere facets -
        the quadrature weights of Eq. (10).
    """

    normals: np.ndarray
    sphere_areas: np.ndarray
    #: Triangulation the normals came from, kept so that Section 3.3's albedo
    #: smoothing can use "the adjacency relations [...] of the octant
    #: triangulation".
    sphere_facets: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.normals = np.ascontiguousarray(self.normals, dtype=float)
        self.sphere_areas = np.ascontiguousarray(self.sphere_areas, dtype=float).ravel()
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True)
        if len(self.normals) != len(self.sphere_areas):
            raise ValueError("normals and sphere_areas must have the same length")

    def __len__(self) -> int:
        return len(self.normals)

    @classmethod
    def from_sphere(cls, n_rows: int = 8) -> FacetGeometry:
        """Facet normals of an octant-triangulated unit sphere (``8 N^2`` of them).

        ``n_rows = 8`` gives 512 normals and ``n_rows = 11`` gives 968, the
        "order 1000" the paper recommends.
        """
        s = octant_triangulation(n_rows)
        body = Polyhedron(s.vertices, s.facets)
        return cls(body.normals, body.areas, s.facets)

    @classmethod
    def from_ellipsoid(cls, a: float, b: float, c: float, n_rows: int = 8) -> FacetGeometry:
        """Facet normals of a triangulated triaxial ellipsoid.

        Section 3.5 warns that this uses prior information: "when the
        directions were those of a triangulated ellipsoid, the best result was
        obtained with the axis ratios that best described the overall
        dimensions of the target.  Such a choice of the surface normals uses a
        priori information that we, in general, do not have."
        """
        s = octant_triangulation(n_rows)
        body = Polyhedron(s.vertices * np.array([a, b, c], dtype=float), s.facets)
        return cls(body.normals, body.areas, s.facets)

    def spherical(self) -> tuple[np.ndarray, np.ndarray]:
        """Polar angle and azimuth of the normals, ``(theta_j, psi_j)``."""
        return unit_to_spherical(self.normals)

    def neighbours(self) -> list[np.ndarray]:
        """Adjacent facets, for the albedo smoothing of Section 3.3.

        Uses the octant triangulation the normals came from when it is known,
        and otherwise falls back to the three nearest normal directions.
        """
        if self.sphere_facets is not None:
            return facet_adjacency(self.sphere_facets)
        tree = cKDTree(self.normals)
        _, idx = tree.query(self.normals, k=4)
        return [np.asarray([j for j in row[1:]], dtype=np.int64) for row in idx]


def nonconvexity_residual(geometry: FacetGeometry, areas: np.ndarray) -> np.ndarray:
    """The vector ``sum_j n_j g_j`` of Eq. (3).

    Section 3.3: "if the sum of the facet vectors (3) is very small, the
    lightcurve features are in all probability caused by the shape.  A strongly
    nonzero residual vector indicates albedo variegation."  Section 3.5 reports
    values of ``|residual| / total area`` "between 0.001 and 0.007" for bodies
    of constant albedo.

    Parameters
    ----------
    geometry:
        The normal directions.
    areas:
        ``(M,)`` solved facet values ``g_j``.

    Returns
    -------
    numpy.ndarray
        ``(3,)`` residual vector; divide by ``areas.sum()`` for the ratio the
        paper quotes.
    """
    return np.asarray(areas, dtype=float) @ geometry.normals


def ellipsoid_log_curvature(
    geometry: FacetGeometry, a: float, b: float, c: float
) -> np.ndarray:
    """``log G`` of a triaxial ellipsoid, evaluated at the normal directions.

    Section 3.2 proposes exactly this as the starting point: "the initial guess
    for iteration can be, e.g., a suitable triaxial ellipsoid; the logarithm of
    its curvature function (given in KLL) can be fitted by the chosen number of
    coefficients using linear least squares."

    For semi-axes ``a, b, c`` the curvature function in terms of the normal is
    ``G(n) = (abc)^2 / (a^2 n_x^2 + b^2 n_y^2 + c^2 n_z^2)^2``.
    """
    q = (geometry.normals**2) @ np.array([a**2, b**2, c**2], dtype=float)
    return 2.0 * np.log(a * b * c) - 2.0 * np.log(q)


class ConvexModel:
    """Maps facet values ``g`` to brightnesses - the operator ``A`` of Eq. (2).

    Parameters
    ----------
    geometry:
        Normal directions and their sphere areas.
    law:
        Scattering law ``S``.
    albedo:
        ``varpi_j``, scalar or ``(M,)``.  Section 3.1 notes that if the
        convexity constraint Eq. (3) "is omitted, ``g`` is taken to include
        albedo variegation", which is the default here.
    """

    def __init__(
        self,
        geometry: FacetGeometry,
        law: ScatteringLaw | None = None,
        albedo: np.ndarray | float = 1.0,
    ) -> None:
        self.geometry = geometry
        self.law = law if law is not None else LommelSeeligerLambert(0.1)
        self.albedo = np.broadcast_to(
            np.asarray(albedo, dtype=float), (len(geometry),)
        ).copy()

    def design_matrix(
        self,
        earth_body: np.ndarray,
        sun_body: np.ndarray,
        alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        """Eq. (4) - the ``(N, M)`` matrix ``A``.

        Parameters
        ----------
        earth_body, sun_body:
            ``(N, 3)`` unit vectors ``E_i`` and ``E0_i`` in the body frame.
        alpha:
            ``(N,)`` solar phase angles, needed only by phase-dependent laws.

        Returns
        -------
        numpy.ndarray
            ``A``, zero wherever ``mu <= 0`` or ``mu0 <= 0``.
        """
        n = self.geometry.normals
        mu = np.asarray(earth_body, dtype=float) @ n.T
        mu0 = np.asarray(sun_body, dtype=float) @ n.T
        if self.law.uses_phase_angle:
            if alpha is None:
                raise ValueError("this scattering law needs the phase angles")
            a = np.asarray(alpha, dtype=float)[:, None]
        else:
            a = None
        return self.law(mu, mu0, a) * self.albedo

    def brightness(self, areas: np.ndarray, design: np.ndarray) -> np.ndarray:
        """``L = A g``."""
        return design @ np.asarray(areas, dtype=float)

    def polyhedron(self, areas: np.ndarray, **kwargs) -> MinkowskiResult:
        """Recover the body from ``g`` by Minkowski minimisation (Appendix C)."""
        return minkowski_solve(self.geometry.normals, areas, **kwargs)


@dataclass
class InversionResult:
    """Outcome of a convex inversion.

    Attributes
    ----------
    areas:
        ``(M,)`` solved facet values ``g_j``.
    parameters:
        The raw optimisation parameters (``a_j`` or ``a_lm``).
    spin:
        The rotation state used, including any fitted pole and period.
    law:
        The scattering law used, including any fitted parameters.
    chi2:
        Final objective value, *excluding* the convexity-regularisation rows.
    rms:
        ``sqrt(chi2 / (N - 3))``, comparable with the reference
        implementation's ``dev``.
    n_iterations, success, message:
        Optimiser diagnostics.
    nonconvexity:
        ``|sum_j n_j g_j| / sum_j g_j`` - Section 3.3's indicator.
    model_lightcurves:
        Per-curve model brightnesses, normalised the same way as the data.
    """

    areas: np.ndarray
    parameters: np.ndarray
    spin: SpinState
    law: ScatteringLaw
    chi2: float
    rms: float
    n_iterations: int
    success: bool
    message: str
    nonconvexity: float
    model_lightcurves: list[np.ndarray] = field(default_factory=list)

    def shape(self, geometry: FacetGeometry, **kwargs) -> MinkowskiResult:
        """Run Minkowski minimisation on :attr:`areas`."""
        return minkowski_solve(geometry.normals, self.areas, **kwargs)


class _ConvexInversionBase:
    """Shared machinery for the two Section 3 parametrisations."""

    def __init__(
        self,
        data: LightcurveSet,
        geometry: FacetGeometry,
        spin: SpinState,
        law: ScatteringLaw | None = None,
        objective: Objective | str = Objective.RELATIVE,
        convexity_weight: float = 0.1,
        convexity_components: str = "xyz",
        albedo: np.ndarray | float = 1.0,
    ) -> None:
        if len(data) == 0:
            raise ValueError("need at least one lightcurve")
        self.data = data
        self.geometry = geometry
        self.spin = spin
        self.model = ConvexModel(geometry, law, albedo)
        self.objective = Objective(objective)
        self.convexity_weight = float(convexity_weight)
        if convexity_components not in ("xyz", "z", "none"):
            raise ValueError("convexity_components must be 'xyz', 'z' or 'none'")
        self.convexity_components = convexity_components

        self._counts = data.counts
        self._offsets = data.offsets
        self._observed = self._build_observed()
        # sqrt of the per-curve weight, broadcast to every point of that curve.
        self._point_weight = np.concatenate(
            [np.full(len(c), np.sqrt(max(c.weight, 0.0))) for c in data]
        )
        self._design_cache: tuple[tuple, np.ndarray] | None = None

    # ------------------------------------------------------------------
    def _build_observed(self) -> np.ndarray:
        """The observed side, normalised per the chosen objective."""
        if self.objective is Objective.ABSOLUTE:
            return np.concatenate([c.brightness for c in self.data])
        return np.concatenate([c.normalised for c in self.data])

    def _design(self, spin: SpinState, law: ScatteringLaw) -> np.ndarray:
        key = (
            spin.lam, spin.beta, spin.period, spin.t0, spin.phi0, spin.yorp,
            tuple(np.atleast_1d(law.parameters).tolist()), type(law).__name__,
        )
        if self._design_cache is not None and self._design_cache[0] == key:
            return self._design_cache[1]
        sun, earth = self.data.body_directions(spin)
        alpha = self.data.phase_angles if law.uses_phase_angle else None
        self.model.law = law
        design = self.model.design_matrix(earth, sun, alpha)
        self._design_cache = (key, design)
        return design

    def _model_and_jacobian(
        self, areas: np.ndarray, design: np.ndarray, want_jac: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Normalised model values and ``d(model)/dg`` for the chosen objective."""
        raw = design @ areas
        if self.objective is Objective.ABSOLUTE:
            return raw, (design if want_jac else None)

        out = np.empty_like(raw)
        jac = np.empty_like(design) if want_jac else None
        for i, c in enumerate(self.data):
            lo, hi = self._offsets[i], self._offsets[i + 1]
            block = raw[lo:hi]
            if self.objective is Objective.RENORMALISED:
                # Eq. (7): divide by the *observed* mean only.
                scale = c.mean_brightness
                out[lo:hi] = block / scale
                if want_jac:
                    jac[lo:hi] = design[lo:hi] / scale
            else:
                # Eq. (13): divide by the *model's* own mean, so the scale of
                # each lightcurve drops out entirely.
                mean = block.mean()
                if mean <= 0:
                    mean = 1e-300
                y = block / mean
                out[lo:hi] = y
                if want_jac:
                    col_mean = design[lo:hi].mean(axis=0)
                    jac[lo:hi] = (design[lo:hi] - np.outer(y, col_mean)) / mean
        return out, jac

    def _convexity_rows(self, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eq. (3) as extra residuals and their derivative with respect to ``g``.

        Section 3.3: "convexity can be enforced in practice by adding the square
        of the length of the vector sum to chi^2 as a regularization function.
        This is equivalent to modifying the original chi^2 of (5) by adding
        three zero elements to L and three new rows to A."  Section 3.4 notes
        that with Eq. (13) "it is sufficient to include only the z-component
        term of (3)", which is ``convexity_components="z"``.
        """
        if self.convexity_components == "none" or self.convexity_weight == 0.0:
            return np.zeros(0), np.zeros((0, len(areas)))
        n = self.geometry.normals
        cols = slice(None) if self.convexity_components == "xyz" else slice(2, 3)
        rows = (self.convexity_weight * n.T[cols])  # (k, M)
        return rows @ areas, rows

    def _residuals(self, areas: np.ndarray, design: np.ndarray, want_jac: bool):
        model, jac = self._model_and_jacobian(areas, design, want_jac)
        res = (model - self._observed) * self._point_weight
        if want_jac and jac is not None:
            jac = jac * self._point_weight[:, None]
        creg, cjac = self._convexity_rows(areas)
        full = np.concatenate([res, creg])
        if not want_jac:
            return full, None
        return full, np.vstack([jac, cjac])

    def _finalise(
        self,
        areas: np.ndarray,
        params: np.ndarray,
        spin: SpinState,
        law: ScatteringLaw,
        n_iter: int,
        success: bool,
        message: str,
    ) -> InversionResult:
        design = self._design(spin, law)
        model, _ = self._model_and_jacobian(areas, design, False)
        res = model - self._observed
        chi2 = float(res @ res)
        n = len(res)
        curves = [model[self._offsets[i] : self._offsets[i + 1]] for i in range(len(self.data))]
        residual = nonconvexity_residual(self.geometry, areas)
        return InversionResult(
            areas=areas,
            parameters=params,
            spin=spin.normalised(),
            law=law,
            chi2=chi2,
            rms=float(np.sqrt(chi2 / max(n - 3, 1))),
            n_iterations=n_iter,
            success=success,
            message=message,
            nonconvexity=float(np.linalg.norm(residual) / max(areas.sum(), 1e-300)),
            model_lightcurves=curves,
        )


class HarmonicInversion(_ConvexInversionBase):
    """Section 3.2 - exponential spherical-harmonics series, Levenberg-Marquardt.

    .. math::  G(\\vartheta, \\psi) = \\exp\\left(\\sum_{lm} a_{lm} Y_{lm}\\right)

    discretised by Eq. (10), ``g_j = G(vartheta_j, psi_j) Delta sigma_j``.

        Since we are minimizing a nonlinear least-squares function (5) and the
        number of the coefficients a_lm to be solved for is not large
        (typically from, say, 40 to 100), it is advantageous to use the
        Levenberg-Marquardt optimization scheme.

    Section 3.2 also records why this is the method to start with: "the
    optimization procedure usually converges very efficiently toward the
    correct solution even with a poor initial guess [...] the robust
    convergence is usually retained when any fixed parameters (e.g., period,
    pole, or scattering law) are changed to free ones."

    Parameters
    ----------
    data:
        The observations.
    geometry:
        Normal directions.
    spin:
        Initial rotation state.
    lmax:
        Truncation degree; ``6`` gives 49 coefficients.
    law:
        Initial scattering law.
    objective, convexity_weight, convexity_components, albedo:
        As in :class:`_ConvexInversionBase`.
    fit_pole:
        Free the pole ``(lambda, beta)``.
    fit_period:
        Free the sidereal period.
    fit_scattering:
        Free the scattering-law parameters.
    """

    def __init__(
        self,
        data: LightcurveSet,
        geometry: FacetGeometry,
        spin: SpinState,
        lmax: int = 6,
        law: ScatteringLaw | None = None,
        objective: Objective | str = Objective.RELATIVE,
        convexity_weight: float = 0.1,
        convexity_components: str = "xyz",
        albedo: np.ndarray | float = 1.0,
        fit_pole: bool = False,
        fit_period: bool = False,
        fit_scattering: bool = False,
    ) -> None:
        super().__init__(
            data, geometry, spin, law, objective, convexity_weight,
            convexity_components, albedo,
        )
        self.lmax = int(lmax)
        theta, psi = geometry.spherical()
        self.basis = design_matrix(self.lmax, theta, psi)  # (M, K)
        self.fit_pole = bool(fit_pole)
        self.fit_period = bool(fit_period)
        self.fit_scattering = bool(fit_scattering)
        # Only vary the scattering parameters the objective can actually
        # constrain; see ScatteringLaw.free_parameter_mask.
        self._law_mask = np.asarray(self.model.law.free_parameter_mask, dtype=bool)
        _lo, _hi = self.model.law.parameter_bounds
        self._law_bounds = (
            np.asarray(_lo, float)[self._law_mask],
            np.asarray(_hi, float)[self._law_mask],
        )
        # Section 3.4: with relative photometry "the coefficient a00 in (8) is
        # a scale factor as well, so it can be left out of the parameter set".
        self.fix_scale = self.objective is Objective.RELATIVE
        self._fixed_a00 = 0.0

    @property
    def n_coefficients(self) -> int:
        """Number of harmonic coefficients, ``(lmax + 1)^2``."""
        return n_coefficients(self.lmax)

    def initial_coefficients(self, a: float = 1.0, b: float = 1.0, c: float = 1.0) -> np.ndarray:
        """Least-squares fit of ``log G`` for an ellipsoid - the paper's start."""
        target = ellipsoid_log_curvature(self.geometry, a, b, c)
        coeffs, *_ = np.linalg.lstsq(self.basis, target, rcond=None)
        return coeffs

    # ------------------------------------------------------------------
    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, SpinState, ScatteringLaw]:
        k = self.n_coefficients
        coeffs = params[:k].copy()
        if self.fix_scale:
            coeffs[0] = self._fixed_a00
        pos = k
        spin, law = self.spin, self.model.law
        if self.fit_pole:
            spin = SpinState(
                params[pos], params[pos + 1], spin.period,
                spin.t0, spin.phi0, spin.yorp,
            )
            pos += 2
        if self.fit_period:
            spin = SpinState(spin.lam, spin.beta, params[pos], spin.t0, spin.phi0, spin.yorp)
            pos += 1
        if self.fit_scattering:
            mask = self._law_mask
            full = np.atleast_1d(law.parameters).astype(float).copy()
            full[mask] = params[pos : pos + int(mask.sum())]
            law = law.with_parameters(full)
            pos += int(mask.sum())
        return coeffs, spin, law

    def areas_from_coefficients(self, coeffs: np.ndarray) -> np.ndarray:
        """Eq. (8) and (10): ``g_j = exp(sum a_lm Y_lm(n_j)) Delta sigma_j``.

        The exponent is clipped before exponentiating.  Eq. (6) makes every
        ``g_j`` positive for *any* coefficients, which is the whole point, but
        it also means a wild trial vector can overflow; clipping keeps such a
        proposal finite so the caller sees a bad chi-squared rather than a NaN.
        """
        return np.exp(np.clip(self.basis @ coeffs, -300.0, 300.0)) * self.geometry.sphere_areas

    def _pack_initial(self, coeffs: np.ndarray) -> np.ndarray:
        parts = [coeffs]
        if self.fit_pole:
            parts.append([self.spin.lam, self.spin.beta])
        if self.fit_period:
            parts.append([self.spin.period])
        if self.fit_scattering:
            parts.append(np.atleast_1d(self.model.law.parameters)[self._law_mask])
        return np.concatenate([np.atleast_1d(np.asarray(p, dtype=float)) for p in parts])

    def _residual_fn(self, params: np.ndarray) -> np.ndarray:
        coeffs, spin, law = self._unpack(params)
        areas = self.areas_from_coefficients(coeffs)
        res, _ = self._residuals(areas, self._design(spin, law), False)
        return res

    def _jacobian_fn(self, params: np.ndarray) -> np.ndarray:
        coeffs, spin, law = self._unpack(params)
        areas = self.areas_from_coefficients(coeffs)
        design = self._design(spin, law)
        _, jac_g = self._residuals(areas, design, True)
        # Chain rule: dg_j/da_lm = g_j Y_lm(n_j).
        jac = jac_g @ (areas[:, None] * self.basis)
        if self.fix_scale:
            jac[:, 0] = 0.0
        extra = []
        if self.fit_pole or self.fit_period or self.fit_scattering:
            # These enter only through A, which has no closed form here, so
            # they are differenced numerically.
            base = self._residual_fn(params)
            k = self.n_coefficients
            for idx in range(k, len(params)):
                step = 1e-6 * max(abs(params[idx]), 1e-3)
                bumped = params.copy()
                bumped[idx] += step
                extra.append((self._residual_fn(bumped) - base) / step)
        if extra:
            jac = np.column_stack([jac, np.column_stack(extra)])
        return jac

    def run(
        self,
        initial: np.ndarray | None = None,
        axes: tuple[float, float, float] = (1.3, 1.0, 0.9),
        max_iter: int = 50,
        verbose: bool = False,
        xtol: float = 1e-10,
        ftol: float = 1e-10,
    ) -> InversionResult:
        """Fit the coefficients.

        Parameters
        ----------
        initial:
            Starting coefficients; defaults to the ellipsoid fit of
            :meth:`initial_coefficients` with semi-axes ``axes``.
        axes:
            Semi-axes of that initial ellipsoid.
        max_iter:
            Maximum Levenberg-Marquardt iterations.
        verbose:
            Print SciPy's iteration report.
        xtol, ftol:
            Convergence tolerances.

        Returns
        -------
        InversionResult
        """
        coeffs0 = (
            self.initial_coefficients(*axes)
            if initial is None
            else np.asarray(initial, dtype=float)
        )
        self._fixed_a00 = float(coeffs0[0])
        p0 = self._pack_initial(coeffs0)

        # Bounds are only needed for the scattering block; everything else is
        # genuinely unconstrained.  SciPy's "lm" cannot take bounds, so the
        # bounded case switches to the trust-region method.
        lower = np.full(len(p0), -np.inf)
        upper = np.full(len(p0), np.inf)
        if self.fit_scattering:
            n_law = int(self._law_mask.sum())
            lower[len(p0) - n_law :] = self._law_bounds[0]
            upper[len(p0) - n_law :] = self._law_bounds[1]
            p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)
        bounded = np.isfinite(lower).any() or np.isfinite(upper).any()

        kwargs = dict(
            jac=self._jacobian_fn,
            max_nfev=max_iter * (len(p0) + 1),
            xtol=xtol,
            ftol=ftol,
            verbose=2 if verbose else 0,
        )
        if bounded:
            sol = least_squares(
                self._residual_fn, p0, method="trf", bounds=(lower, upper), **kwargs
            )
        else:
            sol = least_squares(self._residual_fn, p0, method="lm", **kwargs)
        coeffs, spin, law = self._unpack(sol.x)
        areas = self.areas_from_coefficients(coeffs)
        return self._finalise(
            areas, sol.x, spin, law, int(sol.nfev), bool(sol.success), str(sol.message)
        )


class FacetInversion(_ConvexInversionBase):
    """Section 3.1 - one exponential parameter per facet, conjugate gradients.

    .. math::  g_j = \\exp(a_j)

        Since the number of fitted parameters must be large (of order 1000) to
        make sure that the result does not depend on the directions of the
        surface normals, we use the conjugate gradient method for minimizing
        chi^2.

    Section 3.1 also gives the reason this converges at all: because the
    surfaces of constant ``chi^2`` are convex in ``g``-space "there is one and
    only one vector ``g`` with ``g_j >= 0`` [...] that minimizes ``chi^2``", and
    the exponential map is monotone, so "the smallest ``chi^2`` solution is
    unique in the exponential formalism".

    Section 3.5 recommends using this after :class:`HarmonicInversion`: the
    series method gives "a fast initial solution that can then be enhanced with
    the polyhedron method", and the facet parametrisation is what resolves the
    "large planar areas" that mark concavities.
    """

    def __init__(
        self,
        data: LightcurveSet,
        geometry: FacetGeometry,
        spin: SpinState,
        law: ScatteringLaw | None = None,
        objective: Objective | str = Objective.RELATIVE,
        convexity_weight: float = 0.1,
        convexity_components: str = "xyz",
        albedo: np.ndarray | float = 1.0,
    ) -> None:
        super().__init__(
            data, geometry, spin, law, objective, convexity_weight,
            convexity_components, albedo,
        )

    def _objective_and_gradient(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        areas = np.exp(params)
        design = self._design(self.spin, self.model.law)
        res, jac = self._residuals(areas, design, True)
        # d chi^2 / d a_j = 2 (J^T r)_j g_j, since dg_j/da_j = g_j.
        return float(res @ res), 2.0 * (jac.T @ res) * areas

    def run(
        self,
        initial: np.ndarray | None = None,
        max_iter: int = 500,
        gtol: float = 1e-10,
        verbose: bool = False,
    ) -> InversionResult:
        """Minimise by conjugate gradients.

        Parameters
        ----------
        initial:
            Starting facet values ``g_j`` (not ``a_j``); defaults to a uniform
            sphere.  Pass ``previous.areas`` to polish a
            :class:`HarmonicInversion` result.
        max_iter:
            Maximum conjugate-gradient iterations.
        gtol:
            Gradient-norm convergence threshold.
        verbose:
            Print the optimiser's progress.

        Returns
        -------
        InversionResult
        """
        if initial is None:
            start = np.log(self.geometry.sphere_areas)
        else:
            g0 = np.asarray(initial, dtype=float)
            if len(g0) != len(self.geometry):
                raise ValueError("initial must have one value per facet normal")
            start = np.log(np.maximum(g0, 1e-300))

        sol = minimize(
            self._objective_and_gradient,
            start,
            jac=True,
            method="CG",
            options={"maxiter": max_iter, "gtol": gtol, "disp": bool(verbose)},
        )
        areas = np.exp(sol.x)
        return self._finalise(
            areas, sol.x, self.spin, self.model.law,
            int(sol.nit), bool(sol.success), str(sol.message),
        )
