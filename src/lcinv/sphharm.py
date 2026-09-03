"""Real spherical harmonics ``Y_lm``.

Three of the paper's parametrisations are series in ``Y_lm``:

* Eq. (8), the exponential curvature function
  ``G(theta, psi) = exp(sum_lm a_lm Y_lm(theta, psi))``;
* Eq. (15), the nonconvex radius ``r(theta, phi) = exp(sum_lm c_lm Y_lm)``;
* the albedo regularisation of Section 3.3.

Since ``G`` and ``r`` are real, the *real* harmonics are the natural basis, and
the coefficients are then real numbers that an optimiser can vary directly.
With the Condon-Shortley phase carried by ``P_l^m``,

.. math::
    Y_l^0     &= K_{l0} P_l^0(\\cos\\theta) \\\\
    Y_l^m     &= \\sqrt{2}\\, K_{lm} P_l^m(\\cos\\theta) \\cos(m\\phi), && m > 0\\\\
    Y_l^{-m}  &= \\sqrt{2}\\, K_{lm} P_l^m(\\cos\\theta) \\sin(m\\phi), && m > 0

with ``K_lm = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!)``.  The basis is orthonormal on
the unit sphere.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, lpmv

__all__ = [
    "sph_harm_indices",
    "n_coefficients",
    "real_sph_harm",
    "design_matrix",
]


def sph_harm_indices(lmax: int) -> list[tuple[int, int]]:
    """``(l, m)`` pairs up to degree ``lmax``, ordered by ``l`` then ``m``.

    Returns
    -------
    list of tuple
        ``(lmax + 1) ** 2`` pairs, starting with ``(0, 0)``.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    return [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]


def n_coefficients(lmax: int) -> int:
    """Number of real coefficients for a series truncated at ``lmax``.

    Section 3.2 puts the useful range at "typically from, say, 40 to 100"
    coefficients, i.e. ``lmax`` between 6 and 9; Section 4 truncates the
    nonconvex series (15) "at order and degree four".
    """
    return (lmax + 1) ** 2


def _norm(l: int, m: int) -> float:
    """``K_lm``, computed through log-gammas so high ``l`` does not overflow."""
    am = abs(m)
    log = 0.5 * (
        np.log(2 * l + 1)
        - np.log(4.0 * np.pi)
        + gammaln(l - am + 1)
        - gammaln(l + am + 1)
    )
    return float(np.exp(log))


def real_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Evaluate a single real harmonic ``Y_lm``.

    Parameters
    ----------
    l, m:
        Degree and order, ``|m| <= l``.
    theta:
        Polar angle in radians (0 at the north pole).
    phi:
        Azimuth in radians.

    Returns
    -------
    numpy.ndarray
        ``Y_lm(theta, phi)``, broadcast over the inputs.
    """
    if abs(m) > l:
        raise ValueError("|m| must not exceed l")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    am = abs(m)
    p = lpmv(am, l, np.cos(theta))
    k = _norm(l, m)
    if m == 0:
        return k * p
    if m > 0:
        return np.sqrt(2.0) * k * p * np.cos(am * phi)
    return np.sqrt(2.0) * k * p * np.sin(am * phi)


def design_matrix(lmax: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Matrix ``Y`` with ``Y[i, k] = Y_{l_k m_k}(theta_i, phi_i)``.

    This is the object every series in the paper is written against: the
    exponent of Eq. (8) is ``Y @ a``, and so is that of Eq. (15).  Building it
    once and reusing it makes both the value and the derivative of those
    series a single matrix product.

    Parameters
    ----------
    lmax:
        Maximum degree.
    theta, phi:
        ``(N,)`` polar and azimuthal angles in radians.

    Returns
    -------
    numpy.ndarray
        ``(N, (lmax + 1) ** 2)`` design matrix.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape")
    cos_theta = np.cos(theta)
    out = np.empty((theta.size, n_coefficients(lmax)))
    col = 0
    for l in range(lmax + 1):
        # lpmv is the expensive part, so evaluate each |m| once per degree.
        legendre = {am: lpmv(am, l, cos_theta) for am in range(l + 1)}
        for m in range(-l, l + 1):
            am = abs(m)
            k = _norm(l, m)
            if m == 0:
                out[:, col] = k * legendre[0]
            elif m > 0:
                out[:, col] = np.sqrt(2.0) * k * legendre[am] * np.cos(am * phi)
            else:
                out[:, col] = np.sqrt(2.0) * k * legendre[am] * np.sin(am * phi)
            col += 1
    return out
