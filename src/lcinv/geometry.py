"""Rotation state and the transformation between ecliptic and body frames.

The paper works throughout "in the asteroid's frame of reference" - Eq. (4)
needs ``mu = E_i . n_j`` and ``mu0 = E0_i . n_j`` with the observer and Sun
directions expressed in body coordinates.  Section 1 defers the recovery of
"the sidereal period and the pole direction" to the companion paper, so the
rotation state here is an *input* to the shape inversion; it is nonetheless
written as a fitted-parameter-capable :class:`SpinState` because Section 5
notes that "basically, one should solve for the shape simultaneously with the
period and pole".

The convention is the one used by DAMIT, so that its files can be read
directly:

.. math::
    r_\\mathrm{ecl} = R_z(\\lambda)\\, R_y(90^\\circ - \\beta)\\,
                      R_z\\!\\left(\\varphi_0 + \\frac{2\\pi}{P}(t - t_0)
                      + \\tfrac{1}{2}\\upsilon (t-t_0)^2\\right) r_\\mathrm{ast}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["rot_x", "rot_y", "rot_z", "SpinState", "phase_angle", "spherical_to_unit",
           "unit_to_spherical"]


def rot_x(angle: float | np.ndarray) -> np.ndarray:
    """Rotation matrix about ``x`` by ``angle`` radians (anticlockwise)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float | np.ndarray) -> np.ndarray:
    """Rotation matrix about ``y`` by ``angle`` radians (anticlockwise)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float | np.ndarray) -> np.ndarray:
    """Rotation matrix about ``z`` by ``angle`` radians (anticlockwise)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def spherical_to_unit(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Unit vectors from polar angle and azimuth.

    Parameters
    ----------
    theta:
        Polar angle in radians, zero at the north pole.
    phi:
        Azimuth in radians.

    Returns
    -------
    numpy.ndarray
        ``(..., 3)`` unit vectors, one per input angle pair.
    """
    theta, phi = np.asarray(theta, float), np.asarray(phi, float)
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


def unit_to_spherical(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar angle and azimuth (radians) of unit vectors ``vec``."""
    v = np.asarray(vec, float)
    r = np.linalg.norm(v, axis=-1)
    theta = np.arccos(np.clip(v[..., 2] / np.where(r > 0, r, 1.0), -1.0, 1.0))
    return theta, np.arctan2(v[..., 1], v[..., 0])


@dataclass(frozen=True)
class SpinState:
    """Rotation state of an asteroid.

    Parameters
    ----------
    lam:
        Ecliptic longitude ``lambda`` of the spin axis, degrees (J2000).
    beta:
        Ecliptic latitude ``beta`` of the spin axis, degrees (J2000).
    period:
        Sidereal rotation period ``P``, hours.
    t0:
        Reference epoch, light-time corrected Julian date.
    phi0:
        Rotation angle ``varphi_0`` at ``t0``, degrees.
    yorp:
        Linear change of the rotation rate ``upsilon``, rad/day**2.  Zero for
        almost every model.
    """

    lam: float
    beta: float
    period: float
    t0: float = 0.0
    phi0: float = 0.0
    yorp: float = 0.0

    def rotation_angle(self, jd: np.ndarray | float) -> np.ndarray:
        """Rotation angle ``varphi(t)`` in radians."""
        dt = np.asarray(jd, dtype=float) - self.t0
        omega = 2.0 * np.pi / (self.period / 24.0)  # rad/day
        return np.radians(self.phi0) + omega * dt + 0.5 * self.yorp * dt**2

    def pole_vector(self) -> np.ndarray:
        """Unit spin-axis vector in ecliptic coordinates."""
        lam, beta = np.radians(self.lam), np.radians(self.beta)
        return np.array(
            [np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)]
        )

    def matrix_ast_to_ecl(self, jd: float) -> np.ndarray:
        """The ``3x3`` matrix taking body-frame vectors to ecliptic ones."""
        return (
            rot_z(np.radians(self.lam))
            @ rot_y(np.radians(90.0 - self.beta))
            @ rot_z(float(self.rotation_angle(jd)))
        )

    def to_asteroid_frame(self, vec_ecl: np.ndarray, jd: np.ndarray | float) -> np.ndarray:
        """Rotate ecliptic vectors into the body frame.

        Parameters
        ----------
        vec_ecl:
            ``(3,)`` or ``(N, 3)`` ecliptic vectors.
        jd:
            Scalar epoch, or one epoch per row of ``vec_ecl``.

        Returns
        -------
        numpy.ndarray
            Same shape as ``vec_ecl``, in body coordinates.
        """
        v = np.atleast_2d(np.asarray(vec_ecl, dtype=float))
        jd_arr = np.atleast_1d(np.asarray(jd, dtype=float))
        if jd_arr.size == 1:
            out = v @ self.matrix_ast_to_ecl(float(jd_arr[0]))  # (M R^T)^T == R^-1 v
        else:
            if jd_arr.size != v.shape[0]:
                raise ValueError("jd must be scalar or have one entry per vector")
            # r_ast = Rz(-phi) Ry(-(90-beta)) Rz(-lambda) r_ecl, vectorised over phi.
            fixed = rot_y(-np.radians(90.0 - self.beta)) @ rot_z(-np.radians(self.lam))
            w = v @ fixed.T
            phi = self.rotation_angle(jd_arr)
            c, s = np.cos(phi), np.sin(phi)
            out = np.stack(
                [c * w[:, 0] + s * w[:, 1], -s * w[:, 0] + c * w[:, 1], w[:, 2]], axis=1
            )
        return out.reshape(np.shape(vec_ecl))

    def to_ecliptic_frame(self, vec_ast: np.ndarray, jd: np.ndarray | float) -> np.ndarray:
        """Rotate body-frame vectors into ecliptic coordinates.

        The inverse of :meth:`to_asteroid_frame`.

        Parameters
        ----------
        vec_ast:
            ``(3,)`` or ``(N, 3)`` body-frame vectors.
        jd:
            Scalar epoch, or one epoch per row of ``vec_ast``.

        Returns
        -------
        numpy.ndarray
            Same shape as ``vec_ast``, in ecliptic coordinates.
        """
        v = np.atleast_2d(np.asarray(vec_ast, dtype=float))
        jd_arr = np.atleast_1d(np.asarray(jd, dtype=float))
        if jd_arr.size == 1:
            out = v @ self.matrix_ast_to_ecl(float(jd_arr[0])).T
        else:
            if jd_arr.size != v.shape[0]:
                raise ValueError("jd must be scalar or have one entry per vector")
            phi = self.rotation_angle(jd_arr)
            c, s = np.cos(phi), np.sin(phi)
            w = np.stack(
                [c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1
            )
            fixed = rot_z(np.radians(self.lam)) @ rot_y(np.radians(90.0 - self.beta))
            out = w @ fixed.T
        return out.reshape(np.shape(vec_ast))

    def normalised(self) -> "SpinState":
        """Copy with the pole wrapped into ``lambda in [0, 360)``, ``|beta| <= 90``.

        A fit is smooth in ``(lambda, beta)`` past the poles - ``beta = -93``
        is simply the direction ``beta = -87``, ``lambda + 180`` - so the
        optimiser is allowed to wander there, but a reported pole should be
        canonical.
        """
        lam, beta = float(self.lam), float(self.beta)
        beta = (beta + 180.0) % 360.0 - 180.0     # into (-180, 180]
        if beta > 90.0:
            beta, lam = 180.0 - beta, lam + 180.0
        elif beta < -90.0:
            beta, lam = -180.0 - beta, lam + 180.0
        return SpinState(lam % 360.0, beta, self.period, self.t0, self.phi0, self.yorp)

    @property
    def parameters(self) -> np.ndarray:
        """``[lambda, beta, period]`` - the free rotation parameters."""
        return np.array([self.lam, self.beta, self.period])

    def with_parameters(self, values: np.ndarray) -> SpinState:
        """Copy with ``[lambda, beta, period]`` replaced."""
        v = np.asarray(values, dtype=float)
        if v.size != 3:
            raise ValueError("expected [lambda, beta, period]")
        return SpinState(float(v[0]), float(v[1]), float(v[2]), self.t0, self.phi0, self.yorp)


def phase_angle(sun: np.ndarray, earth: np.ndarray) -> np.ndarray:
    """Solar phase angle ``alpha`` in radians between two direction sets.

    Parameters
    ----------
    sun, earth:
        Asteroid-centric vectors towards the Sun and the Earth; they need not
        be normalised and may be given in any single common frame.

    Returns
    -------
    numpy.ndarray
        ``alpha`` in radians, the angle subtended at the asteroid.  Section
        3.5 stresses that "observing geometries must reach large solar phase
        angles"; this is that angle.
    """
    s = np.atleast_2d(np.asarray(sun, dtype=float))
    e = np.atleast_2d(np.asarray(earth, dtype=float))
    s = s / np.linalg.norm(s, axis=1, keepdims=True)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    alpha = np.arccos(np.clip(np.einsum("ij,ij->i", s, e), -1.0, 1.0))
    return alpha if np.ndim(sun) > 1 or np.ndim(earth) > 1 else alpha[0]
