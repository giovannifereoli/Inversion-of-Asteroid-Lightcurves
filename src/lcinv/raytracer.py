"""Section 2 - lightcurves of nonconvex objects, by ray tracing.

    The ray-tracing procedure is quite simple.  First one checks which
    vertices are above each facet's local horizon and which facets connected
    to these vertices are facing this facet.  These facets are the possible
    local blockers of light, and finding (and labeling) them first as well as
    precomputing their positions with respect to the local horizon makes the
    algorithm very fast.  (The facets for which no vertices appear above the
    local horizon belong to the convex hull.)  [...] The possible visibility
    and illumination of each facet is checked in the same way as in the convex
    case; i.e., both mu and mu0 must be positive.  Those of the possibly
    visible and illuminated facets that do not belong to the convex hull must
    be checked further: if their centroids are blocked by any local
    blocker-facet, their contribution to the total brightness is omitted.

Second-order scattering is neglected, as the paper does: "since second-order
scattering is negligible for low albedos, our code only needs to check which
parts of the surface are visible to both the Earth and the Sun".
"""

from __future__ import annotations

import numpy as np

from .mesh import Polyhedron
from .scattering import ScatteringLaw

try:  # pragma: no cover - exercised only when the extension is installed
    import lcinv_rust as _accel
except ImportError:  # pragma: no cover
    _accel = None

__all__ = ["RayTracer", "hexagonal_facet_samples", "ACCELERATED"]

#: True when the optional Rust kernels (``lcinv-rust``) are available.  They
#: are a drop-in replacement for the NumPy path, not a different algorithm: the
#: test suite asserts the two agree element for element.
ACCELERATED = _accel is not None


def hexagonal_facet_samples(n: int) -> np.ndarray:
    """Barycentric coordinates of ``>= n`` points in a hexagonal arrangement.

    Section 2: "we place a number of test points on each facet (in a
    hexagonal mesh with small random perturbations)".  A triangular lattice of
    ``k`` points per side gives ``k(k+1)/2`` points whose neighbours form the
    hexagonal pattern; the smallest such lattice with at least ``n`` points is
    returned.

    Parameters
    ----------
    n:
        Requested number of test points; ``1`` returns the centroid alone.

    Returns
    -------
    numpy.ndarray
        ``(m, 3)`` barycentric coordinates summing to one, with ``m >= n``.
    """
    if n <= 1:
        return np.full((1, 3), 1.0 / 3.0)
    k = 1
    while k * (k + 1) // 2 < n:
        k += 1
    # Lattice cell centres, which keeps every point strictly inside the facet.
    rows = []
    for i in range(k):
        for j in range(k - i):
            rows.append(((i + 1.0 / 3.0) / k, (j + 1.0 / 3.0) / k))
    ab = np.asarray(rows)
    return np.column_stack([1.0 - ab.sum(axis=1), ab])


class RayTracer:
    """Brightness of a (possibly nonconvex) polyhedron.

    Parameters
    ----------
    body:
        The surface, as a closed :class:`~lcinv.mesh.Polyhedron`.
    n_subpoints:
        Number of test points per facet.  The paper's default is the plain
        centroid check (``1``), which "is quite accurate if there are hundreds
        of facets"; raise it when "any facet represents a large portion of the
        total surface area".
    jitter:
        Fractional random perturbation applied to the test points, the paper's
        "small random perturbations".  Ignored when ``n_subpoints == 1``.
    seed:
        Seed for those perturbations, so a tracer is reproducible.
    tol:
        Relative geometric tolerance, in units of the body's size.
    backend:
        ``"auto"`` uses the Rust kernels when ``lcinv-rust`` is installed and
        NumPy otherwise; ``"python"`` forces the NumPy path, which is what the
        cross-checking tests compare against.

    Attributes
    ----------
    hull_facet_mask:
        ``(F,)`` boolean, true for facets with no vertex above their local
        horizon.  Per Section 2 these "belong to the convex hull" and can
        never be blocked, so they skip the intersection test entirely.
    """

    def __init__(
        self,
        body: Polyhedron,
        n_subpoints: int = 1,
        jitter: float = 0.15,
        seed: int | None = 0,
        tol: float = 1e-9,
        backend: str = "auto",
    ) -> None:
        if backend not in ("auto", "rust", "python"):
            raise ValueError("backend must be 'auto', 'rust' or 'python'")
        if backend == "rust" and _accel is None:
            raise RuntimeError(
                "lcinv-rust is not installed; build it with\n"
                "    maturin develop --release -m src/lcinv_rust/Cargo.toml"
            )
        self.backend = backend
        self.body = body
        self.tol = float(tol)
        self.scale = float(np.abs(body.vertices).max()) or 1.0
        self._eps = self.tol * self.scale
        self._samples = self._build_samples(n_subpoints, jitter, seed)
        self._build_blockers()

    # ------------------------------------------------------------------
    # precomputation
    # ------------------------------------------------------------------
    def _build_samples(self, n: int, jitter: float, seed: int | None) -> np.ndarray:
        bary = hexagonal_facet_samples(n)
        if len(bary) > 1 and jitter > 0.0:
            rng = np.random.default_rng(seed)
            bary = bary + rng.uniform(-jitter, jitter, bary.shape) / len(bary)
            bary = np.clip(bary, 1e-3, None)
            bary /= bary.sum(axis=1, keepdims=True)
        return bary

    def _build_blockers(self) -> None:
        if _accel is not None and self.backend != "python":
            self._build_blockers_rust()
            return
        self._build_blockers_numpy()

    def _build_blockers_rust(self) -> None:
        """Section 2's precomputation, in Rust."""
        body = self.body
        pair_facet, pair_blocker, hull, height = _accel.build_blockers(
            np.ascontiguousarray(body.vertices),
            np.ascontiguousarray(body.facets, dtype=np.int64),
            np.ascontiguousarray(body.normals),
            np.ascontiguousarray(body.facet_centroids),
            self._eps,
        )
        self.hull_facet_mask = hull
        self.blocker_height = height
        self._pair_facet = pair_facet
        self._pair_blocker = pair_blocker
        counts = np.bincount(pair_facet, minlength=len(body.facets))
        self.blocker_ids = np.split(pair_blocker, np.cumsum(counts)[:-1])
        self._finish_blocker_setup()

    def _build_blockers_numpy(self) -> None:
        body = self.body
        verts, facets = body.vertices, body.facets
        normals, centroids = body.normals, body.facet_centroids
        n_f = len(facets)

        # Height of every vertex above every facet's local horizon.  Done in
        # chunks so a large body does not need an (F, V) array all at once.
        offsets = np.einsum("ij,ij->i", normals, centroids)
        blocker_ids: list[np.ndarray] = []
        blocker_height = np.zeros(n_f)
        is_hull = np.zeros(n_f, dtype=bool)

        # vertex -> incident facets, as a flat CSR-style pair list.
        vert_of = facets.ravel()
        face_of = np.repeat(np.arange(n_f), 3)
        order = np.argsort(vert_of, kind="stable")
        vert_sorted, face_sorted = vert_of[order], face_of[order]
        starts = np.searchsorted(vert_sorted, np.arange(len(verts)))
        ends = np.searchsorted(vert_sorted, np.arange(len(verts)), side="right")

        chunk = max(1, int(4e6 // max(len(verts), 1)))
        for lo in range(0, n_f, chunk):
            hi = min(lo + chunk, n_f)
            height = verts @ normals[lo:hi].T - offsets[lo:hi]  # (V, chunk)
            above = height > self._eps
            for local, j in enumerate(range(lo, hi)):
                idx = np.flatnonzero(above[:, local])
                if idx.size == 0:
                    # Section 2: no vertex above the horizon => a hull facet.
                    is_hull[j] = True
                    blocker_ids.append(np.empty(0, dtype=np.int64))
                    continue
                cand = np.unique(
                    np.concatenate([face_sorted[starts[v] : ends[v]] for v in idx])
                )
                cand = cand[cand != j]
                # ... "and which facets connected to these vertices are facing
                # this facet".
                facing = np.einsum(
                    "ij,ij->i", centroids[j] - centroids[cand], normals[cand]
                )
                cand = cand[facing > self._eps]
                blocker_ids.append(cand.astype(np.int64))
                blocker_height[j] = float(height[idx, local].mean())

        self.hull_facet_mask = is_hull
        self.blocker_ids = blocker_ids
        #: Mean height of the above-horizon vertices, for Section 4's
        #: convexity regularisation.
        self.blocker_height = blocker_height

        # Flattened (facet, blocker) pairs for vectorised tracing.
        counts = np.asarray([len(b) for b in blocker_ids])
        self._pair_facet = np.repeat(np.arange(n_f), counts)
        self._pair_blocker = (
            np.concatenate(blocker_ids) if counts.sum() else np.empty(0, dtype=np.int64)
        )
        self._finish_blocker_setup()

    def _finish_blocker_setup(self) -> None:
        """Cache the per-pair triangle data and the nudged test points."""
        body = self.body
        tri = body.vertices[body.facets[self._pair_blocker]]
        self._pv0 = tri[:, 0]
        self._pe1 = tri[:, 1] - tri[:, 0]
        self._pe2 = tri[:, 2] - tri[:, 0]
        self._pn = body.normals[self._pair_blocker]

        # Test-point positions, nudged off the surface so that a facet's own
        # edge-neighbours cannot register as blockers.
        pts = np.einsum("fkj,sk->fsj", body.vertices[body.facets], self._samples)
        pts = pts + 8.0 * self._eps * body.normals[:, None, :]
        self._sample_points = np.ascontiguousarray(pts)  # (F, S, 3)
        self._origins = pts[self._pair_facet]  # (P, S, 3)
        # CSR start offsets, which is what the Rust kernel indexes with.
        counts = np.bincount(self._pair_facet, minlength=len(body.facets))
        self._pair_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        # The Rust kernel looks triangles up by facet id rather than by pair,
        # so it needs one entry per facet instead of one per (facet, blocker).
        tri_all = body.vertices[body.facets]
        self._pv0_by_facet = tri_all[:, 0]
        self._pe1_by_facet = tri_all[:, 1] - tri_all[:, 0]
        self._pe2_by_facet = tri_all[:, 2] - tri_all[:, 0]

    @property
    def n_subpoints(self) -> int:
        """Number of test points used per facet."""
        return self._samples.shape[0]

    @property
    def n_blocker_pairs(self) -> int:
        """Total number of (facet, local blocker) pairs to test."""
        return len(self._pair_facet)

    # ------------------------------------------------------------------
    # tracing
    # ------------------------------------------------------------------
    def _unblocked(self, active: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """``(F, S)`` mask of test points not occluded along ``direction``."""
        n_f, n_s = len(self.body.facets), self.n_subpoints
        free = np.ones((n_f, n_s), dtype=bool)
        if self.n_blocker_pairs == 0:
            return free

        # A ray leaving a closed surface can only be stopped where it re-enters
        # the body, so a blocker must face away from the ray: this is the
        # paper's "only those possible blockers that do not belong to the group
        # of possibly illuminated and visible facets need be included".
        sel = active[self._pair_facet] & (self._pn @ direction < 0.0)
        if not sel.any():
            return free

        v0, e1, e2 = self._pv0[sel], self._pe1[sel], self._pe2[sel]
        orig = self._origins[sel]  # (P', S, 3)

        # Moller-Trumbore, vectorised over pairs and test points.
        pvec = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)

        tvec = orig - v0[:, None, :]
        u = np.einsum("isj,ij->is", tvec, pvec) * inv[:, None]
        qvec = np.cross(tvec, e1[:, None, :])
        v = (qvec @ direction) * inv[:, None]
        t = np.einsum("isj,ij->is", qvec, e2) * inv[:, None]

        hit = (
            ok[:, None]
            & (u >= 0.0)
            & (v >= 0.0)
            & (u + v <= 1.0)
            & (t > self._eps)
        )
        if hit.any():
            # Unbuffered scatter, per (pair, test point), so that several
            # blockers accumulate and partial shadowing is preserved.
            np.logical_and.at(free, self._pair_facet[sel], ~hit)
        return free

    def visible_illuminated_fraction(
        self, earth: np.ndarray, sun: np.ndarray
    ) -> np.ndarray:
        """Fraction of each facet that is both visible and illuminated.

        Parameters
        ----------
        earth, sun:
            Unit vectors ``E`` and ``E0`` towards the observer and the Sun, in
            the body frame.

        Returns
        -------
        numpy.ndarray
            ``(F,)`` values in ``[0, 1]``.  Facets with ``mu <= 0`` or
            ``mu0 <= 0`` give ``0``; convex-hull facets give ``1``; the rest
            give the fraction of test points unshadowed in both directions.
        """
        e = np.asarray(earth, dtype=float)
        e0 = np.asarray(sun, dtype=float)
        e = e / np.linalg.norm(e)
        e0 = e0 / np.linalg.norm(e0)

        normals = self.body.normals
        mu, mu0 = normals @ e, normals @ e0
        potential = (mu > 0.0) & (mu0 > 0.0)
        frac = potential.astype(float)

        # Only non-hull facets need the intersection test.
        active = potential & ~self.hull_facet_mask
        if not active.any():
            return frac
        free = self._unblocked(active, e) & self._unblocked(active, e0)
        frac[active] = free[active].mean(axis=1)
        return frac

    def brightness(
        self,
        earth: np.ndarray,
        sun: np.ndarray,
        law: ScatteringLaw,
        alpha: float | None = None,
    ) -> float:
        """Total brightness ``L`` for one observing geometry.

        Implements Eq. (1) summed over the surface,
        ``L = sum_j S(mu_j, mu0_j) varpi_j A_j f_j`` where ``f_j`` is the
        visible-and-illuminated fraction from
        :meth:`visible_illuminated_fraction`.

        Parameters
        ----------
        earth, sun:
            Unit vectors ``E`` and ``E0`` in the body frame.
        law:
            The scattering law ``S``.
        alpha:
            Solar phase angle in radians; derived from the two vectors when
            omitted, and only needed by phase-dependent laws.

        Returns
        -------
        float
            The total brightness ``L``.
        """
        e = np.asarray(earth, dtype=float)
        e0 = np.asarray(sun, dtype=float)
        e = e / np.linalg.norm(e)
        e0 = e0 / np.linalg.norm(e0)
        if alpha is None and law.uses_phase_angle:
            alpha = float(np.arccos(np.clip(e @ e0, -1.0, 1.0)))
        normals = self.body.normals
        s = law(normals @ e, normals @ e0, alpha)
        frac = self.visible_illuminated_fraction(e, e0)
        return float((s * self.body.albedo * self.body.areas * frac).sum())

    def lightcurve(
        self,
        earth: np.ndarray,
        sun: np.ndarray,
        law: ScatteringLaw,
        alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        """Brightnesses for a sequence of geometries.

        Parameters
        ----------
        earth, sun:
            ``(N, 3)`` body-frame direction vectors.
        law:
            The scattering law.
        alpha:
            Optional ``(N,)`` phase angles; computed from the vectors when
            omitted.

        Returns
        -------
        numpy.ndarray
            ``(N,)`` brightnesses.
        """
        e = np.atleast_2d(np.asarray(earth, dtype=float))
        e0 = np.atleast_2d(np.asarray(sun, dtype=float))
        if len(e) != len(e0):
            raise ValueError("earth and sun must have the same length")
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        e0 = e0 / np.linalg.norm(e0, axis=1, keepdims=True)
        if alpha is None:
            a = np.arccos(np.clip(np.einsum("ij,ij->i", e, e0), -1.0, 1.0))
        else:
            a = np.broadcast_to(np.asarray(alpha, dtype=float), (len(e),))

        if _accel is not None and self.backend != "python":
            frac = self.visible_illuminated_fractions(e, e0)
            normals = self.body.normals
            mu = e @ normals.T
            mu0 = e0 @ normals.T
            s = law(mu, mu0, a[:, None] if law.uses_phase_angle else None)
            weight = self.body.albedo * self.body.areas
            return (s * frac * weight).sum(axis=1)

        return np.asarray(
            [
                self.brightness(e[i], e0[i], law, float(a[i]))
                for i in range(len(e))
            ]
        )

    def visible_illuminated_fractions(
        self, earth: np.ndarray, sun: np.ndarray
    ) -> np.ndarray:
        """``(N, F)`` visible-and-illuminated fractions for many geometries.

        The Rust kernel computes every observation in one call and in parallel;
        without it this falls back to looping
        :meth:`visible_illuminated_fraction`.

        Parameters
        ----------
        earth, sun:
            ``(N, 3)`` body-frame directions; normalised internally.

        Returns
        -------
        numpy.ndarray
            ``(N, F)`` fractions in ``[0, 1]``.
        """
        e = np.atleast_2d(np.asarray(earth, dtype=float))
        e0 = np.atleast_2d(np.asarray(sun, dtype=float))
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        e0 = e0 / np.linalg.norm(e0, axis=1, keepdims=True)
        if _accel is None or self.backend == "python":
            return np.asarray(
                [self.visible_illuminated_fraction(e[i], e0[i]) for i in range(len(e))]
            )
        return _accel.trace_fractions(
            np.ascontiguousarray(e),
            np.ascontiguousarray(e0),
            np.ascontiguousarray(self.body.normals),
            np.ascontiguousarray(self._pv0_by_facet),
            np.ascontiguousarray(self._pe1_by_facet),
            np.ascontiguousarray(self._pe2_by_facet),
            self._sample_points,
            self._pair_start,
            np.ascontiguousarray(self._pair_blocker, dtype=np.int64),
            np.ascontiguousarray(self.hull_facet_mask),
            self._eps,
        )
