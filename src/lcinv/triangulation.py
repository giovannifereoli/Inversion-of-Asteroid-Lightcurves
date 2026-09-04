"""Surface triangulation schemes.

Implements the *octant triangulation* of Appendix A of

    M. Kaasalainen and J. Torppa (2001), "Optimization Methods for Asteroid
    Lightcurve Inversion. I. Shape Determination", Icarus 153, 24-36.

Quoting the appendix:

    The surface is divided into eight octants (according to the coordinate
    axes), each of which is divided into ``N`` horizontal rows usually with
    equal ``pi/2N`` spacing in polar angle.  The first rows from the poles
    toward the equator have no azimuthal points between the octant lines;
    then, for every row closer to the equator, there is one azimuthal point
    more than in the previous row.  The points on each row are evenly spaced
    in azimuthal angle. ... The number of facets is ``8 N**2``, and the number
    of vertices ``4 N**2 + 2``.

Two extra direction sets are provided because Section 3.5 of the paper needs
"evenly distributed surface normals" (of order 1000 of them) for the
polyhedron inversion: :func:`fibonacci_directions` and
:func:`triangulated_ellipsoid`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "facet_adjacency",
    "OctantTriangulation",
    "octant_triangulation",
    "triangulated_ellipsoid",
    "fibonacci_directions",
    "facet_normals_and_areas",
]


@dataclass(frozen=True)
class OctantTriangulation:
    """Vertices and facets of an octant-triangulated unit sphere.

    Attributes
    ----------
    rows:
        Number of horizontal rows per octant, ``N`` in Appendix A.
    vertices:
        ``(4 N**2 + 2, 3)`` array of unit vectors.
    facets:
        ``(8 N**2, 3)`` integer array of vertex indices, counter-clockwise
        when seen from outside the body.
    """

    rows: int
    vertices: np.ndarray
    facets: np.ndarray

    @property
    def n_vertices(self) -> int:
        """Number of vertices, ``4 N^2 + 2`` for ``N`` rows (Appendix A)."""
        return int(self.vertices.shape[0])

    @property
    def n_facets(self) -> int:
        """Number of facets, ``8 N^2`` for ``N`` rows (Appendix A)."""
        return int(self.facets.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OctantTriangulation(rows={self.rows}, "
            f"vertices={self.n_vertices}, facets={self.n_facets})"
        )


def _row_directions(theta: float, ring: int) -> np.ndarray:
    """Unit vectors of one horizontal row at polar angle ``theta``.

    ``ring`` is the row number counted from the nearest pole; the row carries
    ``4 * ring`` evenly spaced points (``ring == 0`` is the pole itself).
    """
    if ring == 0:
        return np.array([[0.0, 0.0, float(np.cos(theta))]])
    n_pts = 4 * ring
    phi = 2.0 * np.pi * np.arange(n_pts) / n_pts
    st, ct = np.sin(theta), np.cos(theta)
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.full(n_pts, ct)])


def octant_triangulation(rows: int = 8) -> OctantTriangulation:
    """Build the octant triangulation of the unit sphere (Appendix A).

    Parameters
    ----------
    rows:
        ``N`` in the paper, the number of rows per octant.  "A suitable number
        of rows is typically from 8 to 10."

    Returns
    -------
    OctantTriangulation

    Notes
    -----
    Row ``i`` (counted from the north pole, ``i = 0 ... N``) sits at polar
    angle ``i * pi / (2 N)`` and carries ``4 i`` vertices, so that each octant
    row gains exactly one azimuthal point over its predecessor and the
    triangle count per octant row grows by two.  The southern hemisphere is
    the mirror image.
    """
    if rows < 1:
        raise ValueError("rows must be >= 1")
    n = int(rows)

    verts: list[np.ndarray] = []
    row_index: list[np.ndarray] = []  # vertex indices per row, north pole -> south pole

    def _add_row(dirs: np.ndarray) -> None:
        start = len(verts)
        verts.extend(dirs)
        row_index.append(np.arange(start, start + len(dirs)))

    # North pole down to and including the equator.
    for i in range(n + 1):
        _add_row(_row_directions(i * np.pi / (2 * n), i))
    # Below the equator down to the south pole (the equator is already in).
    for i in range(n - 1, -1, -1):
        _add_row(_row_directions(np.pi - i * np.pi / (2 * n), i))

    vertices = np.asarray(verts, dtype=float)

    facets: list[tuple[int, int, int]] = []
    for k in range(len(row_index) - 1):
        facets.extend(_strip_facets(row_index[k], row_index[k + 1]))
    facets_arr = np.asarray(facets, dtype=np.int64)

    expected_v, expected_f = 4 * n * n + 2, 8 * n * n
    if vertices.shape[0] != expected_v or facets_arr.shape[0] != expected_f:  # pragma: no cover
        raise RuntimeError(
            f"octant triangulation produced {vertices.shape[0]} vertices / "
            f"{facets_arr.shape[0]} facets, expected {expected_v} / {expected_f}"
        )
    return OctantTriangulation(rows=n, vertices=vertices, facets=facets_arr)


def _strip_facets(upper: np.ndarray, lower: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate the strip between two consecutive rows.

    ``upper`` lies closer to the north pole.  One row always holds four points
    more than the other (or is a single pole vertex).  Triangles come out
    counter-clockwise as seen from outside the sphere.
    """
    nu, nl = len(upper), len(lower)
    if nu == nl:  # pragma: no cover - impossible in the octant scheme
        raise ValueError("adjacent octant rows must differ in size")
    wide_is_lower = nl > nu
    wide, narrow = (lower, upper) if wide_is_lower else (upper, lower)
    nw, nn = len(wide), len(narrow)

    tris: list[tuple[int, int, int]] = []
    if nn == 1:  # polar cap: a fan of `nw` triangles
        p = int(narrow[0])
        for k in range(nw):
            a, b = int(wide[k]), int(wide[(k + 1) % nw])
            tris.append((p, a, b) if wide_is_lower else (p, b, a))
        return tris

    per_quad_narrow, per_quad_wide = nn // 4, nw // 4
    for q in range(4):
        # Quadrant point lists, inclusive of both octant boundary points; the
        # wide list holds exactly one point more than the narrow one.
        nar = [int(narrow[(q * per_quad_narrow + t) % nn]) for t in range(per_quad_narrow + 1)]
        wid = [int(wide[(q * per_quad_wide + t) % nw]) for t in range(per_quad_wide + 1)]
        for t in range(len(nar)):
            tris.append((nar[t], wid[t], wid[t + 1]))
        for t in range(len(nar) - 1):
            tris.append((nar[t], wid[t + 1], nar[t + 1]))

    if wide_is_lower:
        return tris
    return [(a, c, b) for (a, b, c) in tris]


def triangulated_ellipsoid(
    rows: int = 8, axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Octant-triangulated triaxial ellipsoid.

    Section 3.1: "we typically use the facet normals of a sphere or a triaxial
    ellipsoid triangulated in the standard manner".

    Parameters
    ----------
    rows:
        ``N`` of the octant scheme.
    axes:
        Semi-axes ``(a, b, c)``.

    Returns
    -------
    vertices, facets
    """
    tri = octant_triangulation(rows)
    verts = tri.vertices * np.asarray(axes, dtype=float)[None, :]
    return verts, tri.facets


def fibonacci_directions(n: int) -> np.ndarray:
    """``n`` near-uniformly distributed unit vectors (spherical Fibonacci lattice).

    Section 3.5 notes that "the number of parameters should be of order 1000
    (corresponding to evenly distributed surface normals) to make the result
    independent of the exact choice of the normal directions".  The Fibonacci
    lattice gives a more even covering than the octant scheme for an arbitrary
    requested count.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    idx = np.arange(n) + 0.5
    z = 1.0 - 2.0 * idx / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0**0.5) * idx
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def facet_normals_and_areas(
    vertices: np.ndarray, facets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outward unit normals, areas and centroids of triangular facets.

    Parameters
    ----------
    vertices:
        ``(V, 3)`` vertex coordinates.
    facets:
        ``(F, 3)`` vertex indices, counter-clockwise seen from outside.

    Returns
    -------
    normals : ``(F, 3)`` unit vectors
    areas : ``(F,)``
    centroids : ``(F, 3)``
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(facets, dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norm
    safe = np.where(norm > 0, norm, 1.0)
    return cross / safe[:, None], areas, (a + b + c) / 3.0


def facet_adjacency(facets: np.ndarray) -> list[np.ndarray]:
    """Facets sharing an edge with each facet.

    Section 3.3 needs these for the albedo smoothing term
    ``f(varpi) = sum_j sum_i (varpi_ij / varpi_j - 1)^2``, where "the
    adjacency relations are not known at this stage, but a very good
    approximation is to use those of the octant triangulation".

    Parameters
    ----------
    facets:
        ``(F, 3)`` vertex indices.

    Returns
    -------
    list of numpy.ndarray
        For each facet, the indices of the facets sharing one of its edges
        (normally three, fewer on an open surface).
    """
    f = np.asarray(facets, dtype=np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    owner = np.tile(np.arange(len(f)), 3)
    keys = np.sort(edges, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, owner = keys[order], owner[order]

    neighbours: list[list[int]] = [[] for _ in range(len(f))]
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or not np.array_equal(keys[i], keys[start]):
            group = owner[start:i]
            for a in group:
                neighbours[a].extend(int(b) for b in group if b != a)
            start = i
    return [np.asarray(sorted(set(n)), dtype=np.int64) for n in neighbours]
