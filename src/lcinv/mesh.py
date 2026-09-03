"""Triangulated polyhedra.

Section 2 requires that "the surface must be given as a polyhedron with
triangles as facets", and both the direct problem (ray tracing) and the
inversion results (Minkowski minimisation, Appendix C) are expressed in these
terms.  :class:`Polyhedron` is the common currency of the package.

Facets are stored counter-clockwise as seen from *outside* the body, which is
the convention of DAMIT's ``shape.txt`` and of Wavefront ``.obj``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .convexhull import convex_hull

__all__ = ["Polyhedron"]


class Polyhedron:
    """A closed triangulated surface.

    Parameters
    ----------
    vertices:
        ``(V, 3)`` vertex coordinates.
    facets:
        ``(F, 3)`` vertex indices, counter-clockwise seen from outside.
    albedo:
        Optional per-facet albedo ``varpi`` of Eq. (1); defaults to ``1``.
    validate:
        Check that indices are in range and that no facet is degenerate.
    """

    __slots__ = ("_vertices", "_facets", "_albedo", "_cache")

    def __init__(
        self,
        vertices: np.ndarray,
        facets: np.ndarray,
        albedo: np.ndarray | float | None = None,
        validate: bool = True,
    ) -> None:
        self._vertices = np.ascontiguousarray(vertices, dtype=float)
        self._facets = np.ascontiguousarray(facets, dtype=np.int64)
        if self._vertices.ndim != 2 or self._vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if self._facets.ndim != 2 or self._facets.shape[1] != 3:
            raise ValueError("facets must have shape (F, 3)")
        if albedo is None:
            self._albedo = np.ones(len(self._facets))
        else:
            self._albedo = np.broadcast_to(
                np.asarray(albedo, dtype=float), (len(self._facets),)
            ).copy()
        self._cache: dict[str, np.ndarray] = {}
        if validate:
            self._validate()

    def _validate(self) -> None:
        if len(self._facets) and (
            self._facets.min() < 0 or self._facets.max() >= len(self._vertices)
        ):
            raise ValueError("facet references a vertex index out of range")
        if np.any(self.areas <= 0.0):
            bad = int(np.argmin(self.areas))
            raise ValueError(f"facet {bad} is degenerate (zero area)")

    # ------------------------------------------------------------------
    # basic accessors
    # ------------------------------------------------------------------
    @property
    def vertices(self) -> np.ndarray:
        """``(V, 3)`` vertex coordinates (read-only view)."""
        return self._vertices

    @property
    def facets(self) -> np.ndarray:
        """``(F, 3)`` vertex indices (read-only view)."""
        return self._facets

    @property
    def albedo(self) -> np.ndarray:
        """``(F,)`` per-facet albedo ``varpi``."""
        return self._albedo

    @albedo.setter
    def albedo(self, value: np.ndarray | float) -> None:
        self._albedo = np.broadcast_to(
            np.asarray(value, dtype=float), (len(self._facets),)
        ).copy()

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

    @property
    def n_facets(self) -> int:
        return len(self._facets)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Polyhedron({self.n_vertices} vertices, {self.n_facets} facets)"

    # ------------------------------------------------------------------
    # cached geometry
    # ------------------------------------------------------------------
    def _triangles(self) -> np.ndarray:
        return self._vertices[self._facets]

    @property
    def face_vectors(self) -> np.ndarray:
        """``(F, 3)`` outward normals scaled to twice the facet area."""
        if "fv" not in self._cache:
            t = self._triangles()
            self._cache["fv"] = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return self._cache["fv"]

    @property
    def areas(self) -> np.ndarray:
        """``(F,)`` facet areas - the ``s_j`` of Eq. (11)."""
        if "areas" not in self._cache:
            self._cache["areas"] = 0.5 * np.linalg.norm(self.face_vectors, axis=1)
        return self._cache["areas"]

    @property
    def normals(self) -> np.ndarray:
        """``(F, 3)`` outward unit normals - the ``n_j`` of Eq. (3)."""
        if "normals" not in self._cache:
            self._cache["normals"] = self.face_vectors / (
                2.0 * self.areas[:, None]
            )
        return self._cache["normals"]

    @property
    def facet_centroids(self) -> np.ndarray:
        """``(F, 3)`` facet centroids."""
        if "fc" not in self._cache:
            self._cache["fc"] = self._triangles().mean(axis=1)
        return self._cache["fc"]

    @property
    def volume(self) -> float:
        """Enclosed volume, ``V = (1/6) sum v0 . (v1 x v2)``."""
        t = self._triangles()
        return float(np.einsum("ij,ij->i", t[:, 0], self.face_vectors).sum() / 6.0)

    @property
    def surface_area(self) -> float:
        """Total surface area."""
        return float(self.areas.sum())

    @property
    def centroid(self) -> np.ndarray:
        """Centre of mass of the enclosed solid (uniform density)."""
        t = self._triangles()
        # Each tetrahedron (origin, v0, v1, v2) has volume det/6 and centroid
        # (v0+v1+v2)/4.
        det = np.einsum("ij,ij->i", t[:, 0], self.face_vectors)
        vol = det.sum()
        if abs(vol) < 1e-300:  # pragma: no cover - degenerate body
            return self._vertices.mean(axis=0)
        return (det[:, None] * t.sum(axis=1) / 4.0).sum(axis=0) / vol

    @property
    def equivalent_diameter(self) -> float:
        """Diameter of the sphere of the same volume - DAMIT's ``D``."""
        return float(2.0 * (3.0 * abs(self.volume) / (4.0 * np.pi)) ** (1.0 / 3.0))

    @property
    def facet_normal_sum(self) -> np.ndarray:
        """``sum_j A_j n_j``, which vanishes for any closed surface.

        This is the closure identity behind the convexity constraint Eq. (3);
        see :func:`lcinv.convex.nonconvexity_residual` for the inversion-side
        quantity.
        """
        return 0.5 * self.face_vectors.sum(axis=0)

    # ------------------------------------------------------------------
    # transforms
    # ------------------------------------------------------------------
    def _replace(self, vertices: np.ndarray) -> Polyhedron:
        return Polyhedron(vertices, self._facets, self._albedo, validate=False)

    def translated(self, shift: np.ndarray) -> Polyhedron:
        """Copy translated by ``shift``."""
        return self._replace(self._vertices + np.asarray(shift, dtype=float))

    def scaled(self, factor: float) -> Polyhedron:
        """Copy scaled about the origin by ``factor``."""
        return self._replace(self._vertices * float(factor))

    def rotated(self, matrix: np.ndarray) -> Polyhedron:
        """Copy rotated by the ``3x3`` ``matrix``."""
        return self._replace(self._vertices @ np.asarray(matrix, dtype=float).T)

    def centered(self) -> Polyhedron:
        """Copy with the centre of mass at the origin.

        Appendix C notes that "it is useful to shift the centroid of the
        polyhedron to the origin at each iteration step".
        """
        return self.translated(-self.centroid)

    def to_unit_volume(self) -> Polyhedron:
        """Copy scaled to unit volume - DAMIT's uncalibrated convention."""
        v = abs(self.volume)
        if v <= 0:  # pragma: no cover - degenerate body
            raise ValueError("cannot normalise a body of zero volume")
        return self.scaled(v ** (-1.0 / 3.0))

    def flipped(self) -> Polyhedron:
        """Copy with every facet's winding reversed."""
        return Polyhedron(
            self._vertices, self._facets[:, ::-1], self._albedo, validate=False
        )

    def oriented_outward(self) -> Polyhedron:
        """Copy whose facets wind counter-clockwise seen from outside."""
        return self.flipped() if self.volume < 0 else self

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    def convex_hull(self, method: str = "auto") -> Polyhedron:
        """The convex hull, as a :class:`Polyhedron`.

        Section 2: "For each nonconvex shape we also computed the
        corresponding convex hull (as a polyhedron) from the vertices of the
        triangles on the surface."  ``method`` is passed to
        :func:`lcinv.convexhull.convex_hull`; use ``"giftwrap"`` for the
        Appendix B algorithm.
        """
        hull = convex_hull(self._vertices, method=method)
        used, inverse = np.unique(hull.simplices, return_inverse=True)
        return Polyhedron(self._vertices[used], inverse.reshape(-1, 3))

    def projected_area(self, direction: np.ndarray) -> float:
        """Area of the shadow cast along ``direction``, for a *convex* body.

        Equal to ``sum_j A_j max(n_j . d, 0)``.  Section 3.5 uses exactly this
        quantity when noting that "the projected area of the convex hull is
        larger than that of the original surface"; for a nonconvex body the
        result is an over-estimate, since overlapping facets are counted
        twice.
        """
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        return float(np.maximum(self.face_vectors @ d, 0.0).sum() / 2.0)

    def inertia_tensor(self) -> np.ndarray:
        """Inertia tensor about the centroid for unit density."""
        t = self._triangles() - self.centroid
        det = np.einsum(
            "ij,ij->i", t[:, 0], np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        )
        tensor = np.zeros((3, 3))
        # Standard tetrahedron covariance, summed over (origin, v0, v1, v2).
        for a in range(3):
            for b in range(3):
                s = (
                    2.0 * np.einsum("ki,ki->k", t[:, :, a], t[:, :, b])
                    + t[:, 0, a] * t[:, 1, b]
                    + t[:, 1, a] * t[:, 0, b]
                    + t[:, 0, a] * t[:, 2, b]
                    + t[:, 2, a] * t[:, 0, b]
                    + t[:, 1, a] * t[:, 2, b]
                    + t[:, 2, a] * t[:, 1, b]
                )
                tensor[a, b] = (det * s).sum() / 120.0
        return np.trace(tensor) * np.eye(3) - tensor

    def principal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Principal moments (ascending) and the matching axis matrix.

        Returns
        -------
        moments:
            ``(3,)`` eigenvalues of :meth:`inertia_tensor`, ascending, so the
            first is the long axis of the body.
        axes:
            ``(3, 3)`` matrix whose *rows* are the principal axes.  Use
            ``body.rotated(axes)`` to bring the body into its principal frame.
        """
        moments, vectors = np.linalg.eigh(self.inertia_tensor())
        axes = vectors.T
        # Right-handed, for use as a rotation.
        if np.linalg.det(axes) < 0:
            axes[2] = -axes[2]
        return moments, axes

    def extents(self) -> np.ndarray:
        """Bounding-box side lengths along the principal axes (descending)."""
        _, axes = self.principal_axes()
        local = self._vertices @ axes.T
        return np.sort(local.max(axis=0) - local.min(axis=0))[::-1]

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @classmethod
    def from_shape_txt(cls, path: str | Path) -> Polyhedron:
        """Read a DAMIT ``shape.txt`` file (1-based facet indices)."""
        values = Path(path).read_text().split()
        n_v, n_f = int(values[0]), int(values[1])
        body = np.asarray(values[2 : 2 + 3 * n_v], dtype=float).reshape(n_v, 3)
        idx = np.asarray(values[2 + 3 * n_v : 2 + 3 * n_v + 3 * n_f], dtype=np.int64)
        return cls(body, idx.reshape(n_f, 3) - 1)

    def to_shape_txt(self, path: str | Path) -> None:
        """Write a DAMIT ``shape.txt`` file."""
        lines = [f"{self.n_vertices} {self.n_facets}"]
        lines += [f"{x: .8f} {y: .8f} {z: .8f}" for x, y, z in self._vertices]
        lines += [f"{a + 1} {b + 1} {c + 1}" for a, b, c in self._facets]
        Path(path).write_text("\n".join(lines) + "\n")

    @classmethod
    def from_obj(cls, path: str | Path) -> Polyhedron:
        """Read a Wavefront ``.obj`` file, triangulating any polygon faces."""
        verts: list[list[float]] = []
        faces: list[tuple[int, int, int]] = []
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                # Entries may be "i", "i/j" or "i/j/k"; negative indices are
                # relative to the end of the vertex list.
                idx = []
                for token in parts[1:]:
                    i = int(token.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                faces += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
        return cls(np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64))

    def to_obj(self, path: str | Path, name: str = "asteroid") -> None:
        """Write a Wavefront ``.obj`` file."""
        lines = [f"o {name}"]
        lines += [f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in self._vertices]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self._facets]
        Path(path).write_text("\n".join(lines) + "\n")
