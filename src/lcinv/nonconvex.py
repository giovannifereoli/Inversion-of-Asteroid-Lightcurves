"""Section 4 - nonconvex inversion.

    This problem is a very demanding one, the main complications being that all
    uniqueness theorems are lost and the parameter space is usually plagued by
    local minima. [...] after some experimenting, we have found it best to use
    a short functional series describing the locations of the vertices of a
    triangulated surface.

Two parametrisations are given.  Eq. (15) moves vertices along spherical
radius directions,

.. math::  r(\\theta, \\varphi) = \\exp\\left(\\sum_{lm} c_{lm} Y_{lm}\\right),

and Eq. (16) uses a horizontal cylindrical system,

.. math::  \\rho(x, \\phi) = \\exp\\left(\\sum_{jk} c_{jk} x^j e^{ik\\phi}\\right),

which Section 5 recommends when "a contact binary [...] may sometimes be
better described by moving vertices along the radius directions of a
horizontal cylindrical coordinate system".

Trial lightcurves come from the ray tracer of Section 2, and the fit is by
Levenberg-Marquardt.  The paper's own warnings are worth repeating: the series
is truncated early ("at order and degree four"), "the initial guess should be
a good one (e.g., the series fitted to a convex inversion result)", and

    Nonconvex features in inversion results are typically more qualitative
    than quantitative: the existence of, say, valleys is indicated, but the
    depths of the valleys are not very precise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .convex import Objective
from .geometry import SpinState, unit_to_spherical
from .lightcurve import LightcurveSet
from .mesh import Polyhedron
from .raytracer import RayTracer
from .scattering import LommelSeeligerLambert, ScatteringLaw
from .sphharm import design_matrix, n_coefficients
from .triangulation import octant_triangulation

__all__ = [
    "NonconvexResult",
    "RadialShapeSeries",
    "CylindricalShapeSeries",
    "NonconvexInversion",
    "facet_radius_derivatives",
    "convexity_penalty",
]


def facet_radius_derivatives(
    body: Polyhedron, earth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The analytic derivatives quoted in Section 4.

        If the length of the radius vector at a given vertex is denoted by
        ``r`` and the corresponding unit vector by ``r^``, we have
        ``dA/dr = r^ . (d x n) / 2`` and
        ``dmu/dr = [r^ . (d x E) / 2 - mu dA/dr] / A``, where ``A`` is the area
        of a seen and illuminated triangle whose corner the vertex is, and
        ``d`` is the vector corresponding to the side of the triangle opposite
        the vertex (pointing in the positive rotation direction).

    Parameters
    ----------
    body:
        The current surface.
    earth:
        Unit vector ``E`` (or ``E0``) in the body frame.

    Returns
    -------
    d_area:
        ``(F, 3)`` derivative of each facet's area with respect to the radius
        of each of its three vertices.
    d_mu:
        ``(F, 3)`` derivative of ``mu = E . n`` likewise.

    Notes
    -----
    These give the gradient of the *unshadowed* contribution of a facet.  The
    shadowing test is piecewise constant in the parameters, so
    :class:`NonconvexInversion` differences the full ray-traced model instead;
    this function is provided because the paper states the formulas explicitly,
    and it is verified against finite differences in the test suite.
    """
    e = np.asarray(earth, dtype=float)
    e = e / np.linalg.norm(e)
    tri = body.vertices[body.facets]
    normals, areas = body.normals, body.areas

    d_area = np.empty((len(tri), 3))
    d_mu = np.empty((len(tri), 3))
    for k in range(3):
        vertex = tri[:, k]
        # Side opposite the vertex, "pointing in the positive rotation
        # direction", i.e. against the facet's counter-clockwise winding.
        side = tri[:, (k + 1) % 3] - tri[:, (k + 2) % 3]
        r = np.linalg.norm(vertex, axis=1)
        r_hat = vertex / np.maximum(r, 1e-300)[:, None]
        da = 0.5 * np.einsum("ij,ij->i", r_hat, np.cross(side, normals))
        dmu_num = 0.5 * np.einsum("ij,ij->i", r_hat, np.cross(side, np.broadcast_to(e, side.shape)))
        d_area[:, k] = da
        d_mu[:, k] = (dmu_num - (normals @ e) * da) / np.maximum(areas, 1e-300)
    return d_area, d_mu


def convexity_penalty(tracer: RayTracer) -> float:
    """Section 4's regularisation term - the area "sunk below" the convex hull.

        A useful method is to minimize the area "sunk below" the convex hull of
        the current result, i.e., to encourage convexity.  The regularization
        term consists of the sum of the areas of the facets not in the convex
        hull, each multiplied by the average "height" of the vertices of
        possible blockers above the local horizon.

    Both ingredients are already computed by the tracer when it labels local
    blockers: :attr:`~lcinv.raytracer.RayTracer.hull_facet_mask` marks the
    facets that *are* on the hull, and
    :attr:`~lcinv.raytracer.RayTracer.blocker_height` holds the mean height.

    Returns
    -------
    float
        Zero for a convex body, growing with the depth of concavities.
    """
    sunk = ~tracer.hull_facet_mask
    return float((tracer.body.areas[sunk] * tracer.blocker_height[sunk]).sum())


class RadialShapeSeries:
    """Eq. (15) - vertex radii as an exponential spherical-harmonics series.

    Parameters
    ----------
    n_rows:
        Octant-triangulation density fixing the vertex directions.  Section 4:
        "we fix a set of directions along which we optimize the ``c_lm`` [...]
        the directions and connections are easiest to form by standard
        triangulation".
    lmax:
        Truncation degree.  The paper uses four, "since the effects of detailed
        nonconvexities are certainly drowned in the noise".
    """

    def __init__(self, n_rows: int = 6, lmax: int = 4) -> None:
        self.mesh = octant_triangulation(n_rows)
        self.lmax = int(lmax)
        theta, phi = unit_to_spherical(self.mesh.vertices)
        self.basis = design_matrix(self.lmax, theta, phi)

    @property
    def n_parameters(self) -> int:
        return n_coefficients(self.lmax)

    #: ``Y_00`` is constant, so ``c_00`` scales the whole body.
    scale_index: int = 0

    def body(self, coeffs: np.ndarray) -> Polyhedron:
        """Build the polyhedron for a coefficient vector."""
        r = np.exp(np.clip(self.basis @ np.asarray(coeffs, dtype=float), -30.0, 30.0))
        return Polyhedron(self.mesh.vertices * r[:, None], self.mesh.facets, validate=False)

    def fit(self, body: Polyhedron) -> np.ndarray:
        """Least-squares coefficients reproducing an existing body's radii.

        This is how Section 4 wants the iteration started - "the series fitted
        to a convex inversion result".
        """
        target, valid = _radii_towards(body, self.mesh.vertices)
        if valid.sum() < self.basis.shape[1]:
            raise ValueError("too few surface hits to fit the series")
        coeffs, *_ = np.linalg.lstsq(
            self.basis[valid], np.log(target[valid]), rcond=None
        )
        return coeffs


class CylindricalShapeSeries:
    """Eq. (16) - cylindrical radius as a series in ``x`` and ``phi``.

    .. math::  \\rho(x, \\phi) = \\exp\\left(\\sum_{jk} c_{jk} x^j e^{ik\\phi}\\right)

    Real coefficients are used, so the ``e^{ik phi}`` of the paper appears as a
    ``cos k phi`` / ``sin k phi`` pair.  Section 4 places the cylinder "along
    the long axis of the body", sets ``rho`` to zero at the two endpoints
    ``x-`` and ``x+``, and notes that the series "is only valid in some given
    interval ``[x1, x2]``", the two end intervals being "the first and last
    rows of the triangulation mesh".

    Parameters
    ----------
    n_x, n_phi:
        Mesh rows along the axis and points around it.
    degree_x:
        Highest power ``j`` of ``x``.
    degree_phi:
        Highest azimuthal order ``k``.
    half_length:
        ``x+ = -x- =`` this value; the endpoints where ``rho`` vanishes.
    """

    def __init__(
        self,
        n_x: int = 12,
        n_phi: int = 16,
        degree_x: int = 3,
        degree_phi: int = 4,
        half_length: float = 1.0,
    ) -> None:
        self.n_x, self.n_phi = int(n_x), int(n_phi)
        self.degree_x, self.degree_phi = int(degree_x), int(degree_phi)
        self.half_length = float(half_length)

        # Interior rows only; the tips are the two apex vertices.
        self.x = np.linspace(-1.0, 1.0, self.n_x + 2)[1:-1] * self.half_length
        self.phi = np.linspace(0.0, 2.0 * np.pi, self.n_phi, endpoint=False)

        xx, pp = np.meshgrid(self.x / self.half_length, self.phi, indexing="ij")
        cols = [xx.ravel() ** j for j in range(self.degree_x + 1)]
        for k in range(1, self.degree_phi + 1):
            for j in range(self.degree_x + 1):
                cols.append(xx.ravel() ** j * np.cos(k * pp.ravel()))
                cols.append(xx.ravel() ** j * np.sin(k * pp.ravel()))
        self.basis = np.column_stack(cols)
        self._facets = self._build_facets()

    @property
    def n_parameters(self) -> int:
        return self.basis.shape[1]

    #: The ``j = k = 0`` term is constant, so it scales the whole body.
    scale_index: int = 0

    def _build_facets(self) -> np.ndarray:
        faces: list[tuple[int, int, int]] = []
        n_p = self.n_phi
        for i in range(self.n_x - 1):
            for k in range(n_p):
                a = i * n_p + k
                b = i * n_p + (k + 1) % n_p
                c = (i + 1) * n_p + k
                d = (i + 1) * n_p + (k + 1) % n_p
                # (+phi, +x) winding gives the outward radial normal.
                faces += [(a, b, d), (a, d, c)]
        tip_lo, tip_hi = self.n_x * n_p, self.n_x * n_p + 1
        for k in range(n_p):
            # Caps run the opposite way round to the side quads, so that every
            # edge of the end rings is traversed once in each direction.
            faces.append((tip_lo, (k + 1) % n_p, k))
            base = (self.n_x - 1) * n_p
            faces.append((tip_hi, base + k, base + (k + 1) % n_p))
        return np.asarray(faces, dtype=np.int64)

    @classmethod
    def from_body(cls, body: Polyhedron, margin: float = 0.02, **kwargs) -> CylindricalShapeSeries:
        """Size the cylinder from an existing body's extent along ``x``.

        Section 4: "``rho`` is set to 0 at the two endpoints ``x-``, ``x+``,
        whose ``x`` values can be obtained from the convex inversion result as
        well".
        """
        half = float(np.abs(body.vertices[:, 0]).max()) * (1.0 + margin)
        return cls(half_length=half, **kwargs)

    def body(self, coeffs: np.ndarray) -> Polyhedron:
        """Build the polyhedron for a coefficient vector."""
        rho = np.exp(np.clip(self.basis @ np.asarray(coeffs, dtype=float), -30.0, 30.0))
        xx = np.repeat(self.x, self.n_phi)
        pp = np.tile(self.phi, self.n_x)
        verts = np.column_stack([xx, rho * np.cos(pp), rho * np.sin(pp)])
        tips = np.array([[-self.half_length, 0.0, 0.0], [self.half_length, 0.0, 0.0]])
        return Polyhedron(np.vstack([verts, tips]), self._facets, validate=False)

    def fit(self, body: Polyhedron) -> np.ndarray:
        """Least-squares coefficients approximating an existing body."""
        xx = np.repeat(self.x, self.n_phi)
        pp = np.tile(self.phi, self.n_x)
        dirs = np.column_stack([np.zeros_like(pp), np.cos(pp), np.sin(pp)])
        origins = np.column_stack([xx, np.zeros_like(xx), np.zeros_like(xx)])
        rho, valid = _radii_towards(body, dirs, origins)
        if valid.sum() < self.basis.shape[1]:
            raise ValueError(
                "too few surface hits to fit the cylindrical series; "
                "check half_length against the body's extent along x"
            )
        coeffs, *_ = np.linalg.lstsq(
            self.basis[valid], np.log(rho[valid]), rcond=None
        )
        return coeffs


def _radii_towards(
    body: Polyhedron, directions: np.ndarray, origins: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each origin to the surface along each direction.

    Returns the distances and a mask of the rays that actually hit; a ray
    launched from the axis of a strongly waisted body can miss entirely, and
    such rows must be dropped from a fit rather than given a fake radius.
    """
    d = np.asarray(directions, dtype=float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    o = np.zeros_like(d) if origins is None else np.asarray(origins, dtype=float)

    tri = body.vertices[body.facets]
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = np.zeros(len(d))
    for i in range(len(d)):
        pv = np.cross(d[i], e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o[i] - v0
        u = np.einsum("ij,ij->i", tv, pv) * inv
        qv = np.cross(tv, e1)
        v = (qv @ d[i]) * inv
        t = np.einsum("ij,ij->i", qv, e2) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (t > 1e-9)
        if hit.any():
            out[i] = t[hit].max()
    valid = out > 0.0
    return np.maximum(out, 1e-9), valid


@dataclass
class NonconvexResult:
    """Outcome of :meth:`NonconvexInversion.run`.

    Attributes
    ----------
    body:
        The fitted nonconvex polyhedron.
    coefficients:
        The solved series coefficients.
    chi2, rms:
        Fit quality, excluding the regularisation term.
    convexity_penalty:
        Final value of :func:`convexity_penalty`.
    n_iterations, success, message:
        Optimiser diagnostics.
    """

    body: Polyhedron
    coefficients: np.ndarray
    chi2: float
    rms: float
    convexity_penalty: float
    n_iterations: int
    success: bool
    message: str
    model_lightcurves: list[np.ndarray] = field(default_factory=list)


class NonconvexInversion:
    """Fit a nonconvex body to lightcurves by ray tracing.

    Parameters
    ----------
    data:
        Observations.
    spin:
        Rotation state (held fixed).
    series:
        :class:`RadialShapeSeries` or :class:`CylindricalShapeSeries`.
    law:
        Scattering law.
    objective:
        Which chi-squared to use; :attr:`~lcinv.convex.Objective.RELATIVE` by
        default, as for convex inversion.
    regularisation:
        Weight of :func:`convexity_penalty`.  Section 4: "it is often necessary
        to employ smoothness regularization to suppress unrealistic surface
        fluctuation and the formation of artificial features".
    n_subpoints:
        Test points per facet in the tracer.
    """

    def __init__(
        self,
        data: LightcurveSet,
        spin: SpinState,
        series: RadialShapeSeries | CylindricalShapeSeries | None = None,
        law: ScatteringLaw | None = None,
        objective: Objective | str = Objective.RELATIVE,
        regularisation: float = 0.0,
        n_subpoints: int = 1,
    ) -> None:
        self.data = data
        self.spin = spin
        self.series = series if series is not None else RadialShapeSeries()
        self.law = law if law is not None else LommelSeeligerLambert(0.1)
        self.objective = Objective(objective)
        self.regularisation = float(regularisation)
        self.n_subpoints = int(n_subpoints)

        # Eq. (13) normalises the model by its own mean, so the overall size
        # of the body cancels out exactly.  Section 3.4 makes the same point
        # for the convex case - "the coefficient a00 in (8) is a scale factor
        # as well, so it can be left out of the parameter set" - and the
        # constant term of Eq. (15) or (16) plays that role here.  Left free,
        # the optimiser wanders down this flat direction until the body
        # collapses to a point.
        self.fix_scale = self.objective is Objective.RELATIVE
        self._fixed_scale = 0.0

        self._offsets = data.offsets
        self._sun, self._earth = data.body_directions(spin)
        self._alpha = data.phase_angles
        if self.objective is Objective.ABSOLUTE:
            self._observed = np.concatenate([c.brightness for c in data])
        else:
            self._observed = np.concatenate([c.normalised for c in data])

    def _apply_scale(self, coeffs: np.ndarray) -> np.ndarray:
        """Restore the frozen scale term, if the objective ignores size."""
        if not self.fix_scale:
            return coeffs
        out = np.array(coeffs, dtype=float, copy=True)
        out[self.series.scale_index] = self._fixed_scale
        return out

    def model_brightness(self, coeffs: np.ndarray) -> tuple[np.ndarray, RayTracer]:
        """Ray-traced brightnesses for a coefficient vector."""
        body = self.series.body(self._apply_scale(coeffs))
        tracer = RayTracer(body, n_subpoints=self.n_subpoints)
        raw = tracer.lightcurve(self._earth, self._sun, self.law, self._alpha)
        return raw, tracer

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        if self.objective is Objective.ABSOLUTE:
            return raw
        out = np.empty_like(raw)
        for i, c in enumerate(self.data):
            lo, hi = self._offsets[i], self._offsets[i + 1]
            block = raw[lo:hi]
            scale = c.mean_brightness if self.objective is Objective.RENORMALISED else block.mean()
            out[lo:hi] = block / (scale if scale > 0 else 1e-300)
        return out

    def _residuals(self, coeffs: np.ndarray) -> np.ndarray:
        raw, tracer = self.model_brightness(coeffs)
        res = self._normalise(raw) - self._observed
        if self.regularisation > 0.0:
            penalty = np.sqrt(self.regularisation * max(convexity_penalty(tracer), 0.0))
            return np.concatenate([res, [penalty]])
        return res

    def run(
        self,
        initial: np.ndarray | None = None,
        initial_body: Polyhedron | None = None,
        max_nfev: int = 400,
        verbose: bool = False,
    ) -> NonconvexResult:
        """Fit the series coefficients.

        Parameters
        ----------
        initial:
            Starting coefficients.
        initial_body:
            Alternatively, a body to fit the series to first - typically the
            convex inversion result, which is what Section 4 recommends.
        max_nfev:
            Maximum residual evaluations.  Each one ray-traces the whole data
            set, so this is the dominant cost.
        verbose:
            Print SciPy's progress.

        Returns
        -------
        NonconvexResult
        """
        if initial is not None:
            start = np.asarray(initial, dtype=float)
        elif initial_body is not None:
            start = self.series.fit(initial_body)
        else:
            start = self.series.fit(
                Polyhedron(
                    octant_triangulation(6).vertices, octant_triangulation(6).facets
                )
            )

        self._fixed_scale = float(start[self.series.scale_index])
        sol = least_squares(
            self._residuals, start, method="lm",
            max_nfev=max_nfev, verbose=2 if verbose else 0,
        )
        raw, tracer = self.model_brightness(sol.x)
        model = self._normalise(raw)
        res = model - self._observed
        chi2 = float(res @ res)
        return NonconvexResult(
            body=tracer.body,
            coefficients=self._apply_scale(sol.x),
            chi2=chi2,
            rms=float(np.sqrt(chi2 / max(len(res) - 3, 1))),
            convexity_penalty=convexity_penalty(tracer),
            n_iterations=int(sol.nfev),
            success=bool(sol.success),
            message=str(sol.message),
            model_lightcurves=[
                model[self._offsets[i] : self._offsets[i + 1]] for i in range(len(self.data))
            ],
        )
