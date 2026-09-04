"""Light-scattering laws.

Equation (1) of Kaasalainen & Torppa (2001) writes the brightness
contribution of a visible and illuminated surface patch ``ds`` as

.. math::  dL = S(\\mu, \\mu_0)\\, \\varpi\\, ds

with ``mu = E . n``, ``mu0 = E0 . n``, ``E`` and ``E0`` the unit vectors
towards the observer and the Sun, ``n`` the outward surface normal and
``varpi`` the albedo.  The paper names two laws explicitly,

    Lambert law, for example, is ``S_L = mu mu0``, while Lommel-Seeliger law
    is ``S_LS = S_L / (mu + mu0)``,

and Section 3.5 states that "the scattering law used in all computations was a
combination of Lommel-Seeliger and Lambert laws with equal weights".

Every law here returns ``0`` wherever ``mu <= 0`` or ``mu0 <= 0``, which is the
"of course" in the paper's remark that ``A_ij`` vanishes in that case.

For real photometry the empirical solar phase function of Kaasalainen et al.
(2001, "Optimization Methods ... II") is needed as well; it is
:class:`PhaseFunction`, and :class:`LommelSeeligerLambert` accepts one.  That
combination is the ``LSM = "LSL"`` model tabulated by DAMIT.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = [
    "ScatteringLaw",
    "Lambert",
    "LommelSeeliger",
    "LommelSeeligerLambert",
    "Hapke",
    "PhaseFunction",
    "make_scattering_law",
]


def _visible(mu: np.ndarray, mu0: np.ndarray) -> np.ndarray:
    """Mask of patches that are both illuminated and visible (``mu, mu0 > 0``)."""
    return (mu > 0.0) & (mu0 > 0.0)


class ScatteringLaw(ABC):
    """Base class for a scattering law ``S(mu, mu0, alpha)``."""

    #: Whether :meth:`__call__` uses the solar phase angle.
    uses_phase_angle: ClassVar[bool] = False

    @abstractmethod
    def __call__(
        self, mu: np.ndarray, mu0: np.ndarray, alpha: np.ndarray | float | None = None
    ) -> np.ndarray:
        """Evaluate ``S`` for arrays of ``mu`` and ``mu0``.

        Parameters
        ----------
        mu, mu0:
            ``E . n`` and ``E0 . n``, broadcastable arrays.
        alpha:
            Solar phase angle in radians; required only by laws with
            :attr:`uses_phase_angle` set.

        Returns
        -------
        numpy.ndarray
            ``S``, zero wherever the patch is not both visible and illuminated.
        """

    @property
    def parameters(self) -> np.ndarray:
        """Free parameters as a flat array (empty when the law is fixed)."""
        return np.zeros(0)

    @property
    def free_parameter_mask(self) -> np.ndarray:
        """Which of :attr:`parameters` are identifiable from lightcurve *shape*.

        Eq. (13) renormalises every lightcurve to mean unity, so any parameter
        that acts as a pure multiplicative scale cancels exactly and cannot be
        recovered.  Fitting one anyway lets an optimiser wander along a flat
        direction.  Laws that have such a parameter mark it ``False`` here, and
        :class:`~lcinv.convex.HarmonicInversion` then holds it fixed.
        """
        return np.ones(len(np.atleast_1d(self.parameters)), dtype=bool)

    @property
    def parameter_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical ``(lower, upper)`` limits for :attr:`parameters`.

        Without these an optimiser will happily return a negative opposition
        surge or a negative surge width, which fit slightly better and mean
        nothing.  Unbounded entries use ``-inf`` / ``+inf``.
        """
        n = len(np.atleast_1d(self.parameters))
        return np.full(n, -np.inf), np.full(n, np.inf)

    def with_parameters(self, values: np.ndarray) -> ScatteringLaw:
        """Return a copy with :attr:`parameters` replaced by ``values``.

        This is what lets Section 5's "additional parameters, for example,
        those of the solar phase function of the scattering law" be "left
        adjustable" during an inversion.
        """
        if len(values) != 0:
            raise ValueError(f"{type(self).__name__} takes no free parameters")
        return self


@dataclass(frozen=True)
class Lambert(ScatteringLaw):
    """Lambert law, ``S_L = mu mu0``."""

    def __call__(self, mu, mu0, alpha=None):
        mu, mu0 = np.asarray(mu, dtype=float), np.asarray(mu0, dtype=float)
        return np.where(_visible(mu, mu0), mu * mu0, 0.0)


@dataclass(frozen=True)
class LommelSeeliger(ScatteringLaw):
    """Lommel-Seeliger law, ``S_LS = mu mu0 / (mu + mu0)``."""

    def __call__(self, mu, mu0, alpha=None):
        mu, mu0 = np.asarray(mu, dtype=float), np.asarray(mu0, dtype=float)
        ok = _visible(mu, mu0)
        denom = np.where(ok, mu + mu0, 1.0)
        return np.where(ok, mu * mu0 / denom, 0.0)


@dataclass(frozen=True)
class PhaseFunction:
    """Empirical solar phase function ``f(alpha) = a exp(-alpha/d) + k alpha + 1``.

    The exponential term is the opposition surge, ``k`` the linear slope.
    DAMIT tabulates ``a``, ``d`` and ``k`` as ``lsm_p2``, ``lsm_p3`` and
    ``lsm_p4`` of an ``LSL`` model.

    Parameters
    ----------
    amplitude:
        ``a``, amplitude of the opposition surge.
    width:
        ``d``, width of the surge in radians.
    slope:
        ``k``, linear slope per radian.
    """

    amplitude: float = 0.5
    width: float = 0.1
    slope: float = -0.5

    def __call__(self, alpha: np.ndarray | float) -> np.ndarray:
        a = np.asarray(alpha, dtype=float)
        if self.width <= 0:
            raise ValueError("phase function width must be positive")
        return self.amplitude * np.exp(-a / self.width) + self.slope * a + 1.0

    @property
    def parameters(self) -> np.ndarray:
        return np.array([self.amplitude, self.width, self.slope])


@dataclass(frozen=True)
class LommelSeeligerLambert(ScatteringLaw):
    """``S = f(alpha) [ mu mu0 / (mu + mu0) + c mu mu0 ]``.

    Parameters
    ----------
    lambert_weight:
        ``c``, the weight of the Lambert term relative to Lommel-Seeliger.
        ``c = 1`` reproduces the "equal weights" combination used throughout
        Section 3.5 of the paper; DAMIT's ``LSL`` models use ``c = lsm_p1``,
        typically ``0.1``.
    phase_function:
        Optional :class:`PhaseFunction`.  Leave it ``None`` for the paper's
        simulations, which fold no phase dependence into ``S``.
    """

    lambert_weight: float = 1.0
    phase_function: PhaseFunction | None = None

    @property
    def uses_phase_angle(self) -> bool:  # type: ignore[override]
        return self.phase_function is not None

    def __call__(self, mu, mu0, alpha=None):
        mu, mu0 = np.asarray(mu, dtype=float), np.asarray(mu0, dtype=float)
        ok = _visible(mu, mu0)
        denom = np.where(ok, mu + mu0, 1.0)
        s = mu * mu0 * (1.0 / denom + self.lambert_weight)
        s = np.where(ok, s, 0.0)
        if self.phase_function is None:
            return s
        if alpha is None:
            raise ValueError("this law needs the solar phase angle `alpha`")
        return s * self.phase_function(alpha)

    @property
    def parameters(self) -> np.ndarray:
        if self.phase_function is None:
            return np.array([self.lambert_weight])
        return np.concatenate([[self.lambert_weight], self.phase_function.parameters])

    @property
    def parameter_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.phase_function is None:
            return np.array([0.0]), np.array([np.inf])
        #            c        a       d        k
        lo = np.array([0.0,   0.0,   1e-3,   -10.0])
        hi = np.array([np.inf, 10.0,  2.0,     0.0])
        return lo, hi

    def with_parameters(self, values: np.ndarray) -> LommelSeeligerLambert:
        values = np.asarray(values, dtype=float)
        if self.phase_function is None:
            if values.size != 1:
                raise ValueError("expected 1 parameter (lambert_weight)")
            return LommelSeeligerLambert(float(values[0]), None)
        if values.size != 4:
            raise ValueError("expected 4 parameters (c, a, d, k)")
        return LommelSeeligerLambert(
            float(values[0]),
            PhaseFunction(float(values[1]), float(values[2]), float(values[3])),
        )


@dataclass(frozen=True)
class Hapke(ScatteringLaw):
    """Hapke bidirectional reflectance, cast into the paper's ``S``.

    DAMIT's ``LSM = "H"`` models are parameterised this way.  The single
    scattering albedo ``w`` carries the albedo, so ``varpi`` should be left at
    ``1`` when this law is used.

    .. math::
        S = \\mu\\, r(i, e, \\alpha), \\qquad
        r = \\frac{w}{4\\pi}\\frac{\\mu_{0e}}{\\mu_{0e} + \\mu_e}
            \\left[(1 + B(\\alpha))p(\\alpha) + H(\\mu_{0e})H(\\mu_e) - 1\\right]
            S(i, e, \\psi)

    with a single Henyey-Greenstein phase function ``p``, the opposition surge
    ``B(alpha) = B0 / (1 + tan(alpha/2)/h)``, and Hapke's rational
    approximation to the Chandrasekhar ``H`` function.

    Parameters
    ----------
    w:
        Single-particle scattering albedo (DAMIT ``lsm_p1``).
    g:
        Henyey-Greenstein asymmetry parameter (``lsm_p2``); negative is
        backscattering.
    b0:
        Opposition-surge amplitude (``lsm_p3``).
    h:
        Opposition-surge width (``lsm_p4``).
    roughness:
        Mean slope angle ``theta_bar`` in **degrees** (``lsm_p5``).  Zero
        disables the correction and recovers the smooth-surface law exactly.

    Notes
    -----
    The macroscopic-roughness correction of Hapke (1984; 1993, chapter 12) is
    applied: the true cosines are replaced by the effective ``mu_e`` and
    ``mu_0e`` of a surface tilted by unresolved slopes, and the result is
    multiplied by the shadowing function ``S(i, e, psi)``.  It needs the
    azimuth ``psi`` between the planes of incidence and emergence, which
    follows from ``mu``, ``mu0`` and the phase angle through
    ``cos psi = (cos alpha - mu mu0) / (sin i sin e)``.

    This is a *geometric* roughness model: it describes sub-facet slopes, not
    the resolved shadowing that :class:`~lcinv.raytracer.RayTracer` computes
    between facets.  The two are complementary and can be used together.
    """

    w: float = 0.3
    g: float = -0.3
    b0: float = 0.0
    h: float = 0.05
    roughness: float = 0.0

    uses_phase_angle: ClassVar[bool] = True

    def _h_function(self, x: np.ndarray) -> np.ndarray:
        gamma = np.sqrt(max(0.0, 1.0 - max(self.w, 0.0)))
        return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma)

    def _roughness_correction(
        self, mu: np.ndarray, mu0: np.ndarray, alpha: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Effective cosines and the shadowing function ``S(i, e, psi)``.

        Returns ``(mu0e, mu_e, shadow)``.  Follows Hapke (1993) eqs. 12.45-12.54,
        which split on whether the incidence or the emergence angle is larger.
        """
        theta = np.radians(self.roughness)
        tan_t = np.tan(theta)
        chi = 1.0 / np.sqrt(1.0 + np.pi * tan_t**2)
        cot_t = 1.0 / tan_t

        # Angles.  Clip away from the exact limb so cotangents stay finite.
        eps = 1e-9
        mu_c = np.clip(mu, eps, 1.0 - eps)
        mu0_c = np.clip(mu0, eps, 1.0 - eps)
        i = np.arccos(mu0_c)
        e = np.arccos(mu_c)
        sin_i, sin_e = np.sin(i), np.sin(e)

        # Azimuth between the planes of incidence and emergence.
        cos_psi = np.clip(
            (np.cos(alpha) - mu_c * mu0_c) / np.maximum(sin_i * sin_e, eps), -1.0, 1.0
        )
        psi = np.arccos(cos_psi)
        half = np.sin(psi / 2.0) ** 2

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            def e1(x):
                return np.exp(-2.0 / np.pi * cot_t / np.maximum(np.tan(x), eps))

            def e2(x):
                return np.exp(-1.0 / np.pi * cot_t**2 / np.maximum(np.tan(x), eps) ** 2)

            e1_i, e1_e = e1(i), e1(e)
            e2_i, e2_e = e2(i), e2(e)

            def eta(x, e1x, e2x):
                return chi * (
                    np.cos(x) + np.sin(x) * tan_t * e2x / np.maximum(2.0 - e1x, eps)
                )

            eta_i, eta_e = eta(i, e1_i, e2_i), eta(e, e1_e, e2_e)

            # i <= e branch
            den_a = np.maximum(2.0 - e1_e - (psi / np.pi) * e1_i, eps)
            mu0e_a = chi * (
                mu0_c + sin_i * tan_t * (cos_psi * e2_e + half * e2_i) / den_a
            )
            mue_a = chi * (mu_c + sin_e * tan_t * (e2_e - half * e2_i) / den_a)

            # i > e branch
            den_b = np.maximum(2.0 - e1_i - (psi / np.pi) * e1_e, eps)
            mu0e_b = chi * (mu0_c + sin_i * tan_t * (e2_i - half * e2_e) / den_b)
            mue_b = chi * (
                mu_c + sin_e * tan_t * (cos_psi * e2_i + half * e2_e) / den_b
            )

            small_i = i <= e
            mu0e = np.where(small_i, mu0e_a, mu0e_b)
            mue = np.where(small_i, mue_a, mue_b)

            f = np.exp(-2.0 * np.tan(np.clip(psi, 0.0, np.pi - eps) / 2.0))
            ref = np.where(small_i, mu0_c / np.maximum(eta_i, eps),
                           mu_c / np.maximum(eta_e, eps))
            shadow = (
                (mue / np.maximum(eta_e, eps))
                * (mu0_c / np.maximum(eta_i, eps))
                * chi
                / np.maximum(1.0 - f + f * chi * ref, eps)
            )

        mu0e = np.nan_to_num(mu0e, nan=0.0, posinf=0.0, neginf=0.0)
        mue = np.nan_to_num(mue, nan=0.0, posinf=0.0, neginf=0.0)
        shadow = np.nan_to_num(shadow, nan=1.0, posinf=1.0, neginf=1.0)
        return mu0e, mue, shadow

    def __call__(self, mu, mu0, alpha=None):
        if alpha is None:
            raise ValueError("the Hapke law needs the solar phase angle `alpha`")
        mu, mu0 = np.asarray(mu, dtype=float), np.asarray(mu0, dtype=float)
        a = np.broadcast_to(np.asarray(alpha, dtype=float), np.broadcast(mu, mu0).shape)
        ok = _visible(mu, mu0)

        if self.roughness > 0.0:
            mu0e, mue, shadow = self._roughness_correction(mu, mu0, a)
        else:
            mu0e, mue, shadow = mu0, mu, 1.0

        denom = np.where(ok, mue + mu0e, 1.0)
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

        p = (1.0 - self.g**2) / np.power(1.0 + 2.0 * self.g * np.cos(a) + self.g**2, 1.5)
        surge = self.b0 / (1.0 + np.tan(np.clip(a, 0.0, np.pi - 1e-9) / 2.0) / self.h)
        core = (1.0 + surge) * p + self._h_function(mu0e) * self._h_function(mue) - 1.0
        s = mu * (self.w / (4.0 * np.pi)) * (mu0e / denom) * core * shadow
        return np.where(ok, s, 0.0)

    @property
    def parameters(self) -> np.ndarray:
        return np.array([self.w, self.g, self.b0, self.h, self.roughness])

    @property
    def free_parameter_mask(self) -> np.ndarray:
        # `w` multiplies the whole law and enters the shape only weakly, through
        # H(x); under Eq. (13) it is effectively a scale factor, so it is held
        # fixed.  Free it explicitly only when fitting absolute photometry.
        return np.array([False, True, True, True, True])

    @property
    def parameter_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        #             w      g      B0     h     theta_bar (deg)
        lo = np.array([1e-4, -0.999,  0.0, 1e-4,  0.0])
        hi = np.array([1.0,    0.999, 5.0, 10.0, 60.0])
        return lo, hi

    def with_parameters(self, values: np.ndarray) -> Hapke:
        v = np.asarray(values, dtype=float)
        if v.size != 5:
            raise ValueError("expected 5 parameters (w, g, b0, h, roughness)")
        return Hapke(*(float(x) for x in v))


def make_scattering_law(
    lsm: str, params: list[float | None] | np.ndarray | None = None
) -> ScatteringLaw:
    """Build a law from a DAMIT ``lsm`` code and its ``lsm_p1 ... lsm_p5``.

    Parameters
    ----------
    lsm:
        ``"LSL"`` (Lambert + Lommel-Seeliger) or ``"H"`` (Hapke).
    params:
        Up to five values; ``None`` entries fall back to the class defaults.

    Returns
    -------
    ScatteringLaw
    """
    p = list(params) if params is not None else []
    p += [None] * (5 - len(p))

    def _f(i: int, default: float) -> float:
        v = p[i]
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    code = lsm.strip().upper()
    if code == "LSL":
        # A DAMIT LSL record without a1/d/k means no phase function was fitted.
        if all(p[i] is None for i in (1, 2, 3)):
            return LommelSeeligerLambert(_f(0, 0.1), None)
        return LommelSeeligerLambert(
            _f(0, 0.1), PhaseFunction(_f(1, 0.5), _f(2, 0.1), _f(3, -0.5))
        )
    if code == "H":
        return Hapke(_f(0, 0.3), _f(1, -0.3), _f(2, 0.0), _f(3, 0.05), _f(4, 0.0))
    raise ValueError(f"unknown light-scattering model {lsm!r}; expected 'LSL' or 'H'")
