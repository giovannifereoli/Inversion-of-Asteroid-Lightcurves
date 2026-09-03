"""Section 3.3 - separating shape from albedo variegation.

Once an inversion has produced facet values ``g_j`` *without* the convexity
constraint, those values contain shape and albedo together.  Section 3.3 then
poses the separation as Eq. (11),

.. math::
    \\chi^2_{\\rm sep} = \\sum_j (g_j - s_j \\varpi_j)^2
        + \\lambda_s \\sum_{i=1}^{3}\\Big[\\sum_j n_j s_j\\Big]^2
        + \\lambda_\\varpi f(\\varpi),

where ``s_j`` is the facet area and ``varpi_j`` its albedo.  The second term
forces the *area* part to be convex through Eq. (3); the third smooths the
albedo.  Albedos are confined to ``[a, b]`` by Eq. (12),

.. math::  \\varpi = a + (b - a)\\frac{e^{c}}{e^{c} + 1},

with ``c`` the unconstrained optimisation variable - the same trick as Eq. (6).

The paper is emphatic about what such a solution means:

    In the above manner we get a solution that fits the lightcurves, is
    convex, and describes the albedo asymmetry rather than the actual
    distribution.  All albedo symmetries are absorbed into the shape solution,
    so the albedo "map" is realistic only if there is a single prominent albedo
    spot on the surface.

and that ``lambda_s`` and ``lambda_varpi`` "are to be kept very small as (11)
must be strongly dominated by the first term".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .convex import FacetGeometry

__all__ = ["AlbedoSeparation", "AlbedoResult", "logistic_albedo", "inverse_logistic_albedo"]


def logistic_albedo(c: np.ndarray, low: float, high: float) -> np.ndarray:
    """Eq. (12) - map unconstrained ``c`` into the albedo interval ``[low, high]``."""
    return low + (high - low) / (1.0 + np.exp(-np.asarray(c, dtype=float)))


def inverse_logistic_albedo(w: np.ndarray, low: float, high: float) -> np.ndarray:
    """Inverse of :func:`logistic_albedo`, for building an initial guess."""
    x = (np.asarray(w, dtype=float) - low) / (high - low)
    x = np.clip(x, 1e-9, 1.0 - 1e-9)
    return np.log(x / (1.0 - x))


@dataclass
class AlbedoResult:
    """Outcome of :meth:`AlbedoSeparation.run`.

    Attributes
    ----------
    facet_areas:
        ``(M,)`` the convex area part ``s_j``.
    albedo:
        ``(M,)`` the albedo part ``varpi_j``.
    residual_nonconvexity:
        ``|sum_j n_j s_j| / sum_j s_j`` for the *separated* area part; this is
        what the convexity term drives towards zero.
    data_misfit:
        ``sum_j (g_j - s_j varpi_j)^2``, the first term of Eq. (11).
    success, message:
        Optimiser diagnostics.
    """

    facet_areas: np.ndarray
    albedo: np.ndarray
    residual_nonconvexity: float
    data_misfit: float
    success: bool
    message: str


class AlbedoSeparation:
    """Split solved facet values into a convex shape and an albedo map.

    Parameters
    ----------
    geometry:
        The normal directions used in the inversion.
    areas:
        ``(M,)`` facet values ``g_j`` from an *unconstrained* inversion, i.e.
        one run with ``convexity_components="none"`` so that ``g`` "is taken to
        include albedo variegation".
    albedo_range:
        ``(a, b)`` of Eq. (12).
    lambda_shape:
        ``lambda_s``, the weight of the convexity term.  Keep it small.
    lambda_albedo:
        ``lambda_varpi``, the weight of the smoothing term.  Keep it small.
    neighbours:
        Facet adjacency for the smoothing term; taken from the geometry's own
        triangulation when omitted.
    """

    def __init__(
        self,
        geometry: FacetGeometry,
        areas: np.ndarray,
        albedo_range: tuple[float, float] = (0.5, 1.5),
        lambda_shape: float = 1e-3,
        lambda_albedo: float = 1e-3,
        neighbours: "list[np.ndarray] | None" = None,
    ) -> None:
        self.geometry = geometry
        self.g = np.asarray(areas, dtype=float).ravel()
        if len(self.g) != len(geometry):
            raise ValueError("areas must have one value per normal")
        self.low, self.high = float(albedo_range[0]), float(albedo_range[1])
        if not self.low < self.high:
            raise ValueError("albedo_range must be increasing")
        self.lambda_shape = float(lambda_shape)
        self.lambda_albedo = float(lambda_albedo)
        self.neighbours = neighbours if neighbours is not None else geometry.neighbours()

        # Flattened adjacency pairs for the smoothing residuals.
        pairs = [(j, int(k)) for j, ns in enumerate(self.neighbours) for k in ns]
        self._pair_j = np.asarray([p[0] for p in pairs], dtype=np.int64)
        self._pair_i = np.asarray([p[1] for p in pairs], dtype=np.int64)

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = len(self.g)
        s = np.exp(params[:m])  # positivity, as in Eq. (6)
        w = logistic_albedo(params[m:], self.low, self.high)
        return s, w

    def _residuals(self, params: np.ndarray) -> np.ndarray:
        s, w = self._unpack(params)
        # First term of Eq. (11).
        misfit = self.g - s * w
        # Second: the convexity constraint Eq. (3) on the area part alone.
        conv = np.sqrt(self.lambda_shape) * (s @ self.geometry.normals)
        # Third: f(varpi) = sum_j sum_i (varpi_ij / varpi_j - 1)^2.
        smooth = np.sqrt(self.lambda_albedo) * (w[self._pair_i] / w[self._pair_j] - 1.0)
        return np.concatenate([misfit, conv, smooth])

    def run(
        self, max_nfev: int = 200, verbose: bool = False
    ) -> AlbedoResult:
        """Minimise Eq. (11).

        Parameters
        ----------
        max_nfev:
            Maximum residual evaluations.
        verbose:
            Print SciPy's progress.

        Returns
        -------
        AlbedoResult
        """
        m = len(self.g)
        start = np.concatenate(
            [np.log(np.maximum(self.g, 1e-300)), np.zeros(m)]
        )
        sol = least_squares(
            self._residuals, start, method="trf",
            max_nfev=max_nfev, verbose=2 if verbose else 0,
        )
        s, w = self._unpack(sol.x)
        misfit = self.g - s * w
        return AlbedoResult(
            facet_areas=s,
            albedo=w,
            residual_nonconvexity=float(
                np.linalg.norm(s @ self.geometry.normals) / max(s.sum(), 1e-300)
            ),
            data_misfit=float(misfit @ misfit),
            success=bool(sol.success),
            message=str(sol.message),
        )
