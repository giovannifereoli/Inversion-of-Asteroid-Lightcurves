"""Convex hulls in three dimensions.

Section 2 of Kaasalainen & Torppa (2001) computes, for every nonconvex test
shape, "the corresponding convex hull (as a polyhedron) from the vertices of
the triangles on the surface":

    Such a procedure is easy to write by systematically comparing each point
    with the rest to find planes such that all the other points lie on one
    side of them; this gift-wrapping principle results in an N**2-algorithm.
    ... Other, more efficient (N log N) methods are available in literature
    and on the Internet; some of them, however, are very complex or not
    completely reliable - N**2 is foolproof and not very much slower in
    absolute time when N is less than, say, 1000.

Both options are provided.  :func:`convex_hull` defaults to Qhull through
:class:`scipy.spatial.ConvexHull` (the "available on the Internet" route) and
falls back to, or can be switched to, the literal Appendix B gift-wrapping
implementation in :func:`gift_wrap_hull`.

Step 8 of Appendix B - "remove those points that do not define new planes,
i.e., those that lie in the plane defined by the adjacent vertices" - is
:meth:`HullResult.merge_coplanar`.  It is what turns the triangulated hull
into the polygonal facets whose "large planar parts forming bridges over the
valleys of the original shapes ... have a key role in convex inversion".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull as _QhullConvexHull
from scipy.spatial import QhullError as _QhullError

__all__ = [
    "HullResult",
    "ConvexHullError",
    "convex_hull",
    "gift_wrap_hull",
    "hull_volume_area",
]


class ConvexHullError(ValueError):
    """Raised when a point cloud is too degenerate for a hull to be built.

    Subclasses :class:`ValueError` so callers that merely need to reject a bad
    trial geometry - the Minkowski line search, for one - can catch it without
    depending on SciPy's exception types.
    """


@dataclass
class HullResult:
    """A convex hull as a triangulated polyhedron.

    Attributes
    ----------
    points:
        ``(N, 3)`` array of the input points.
    vertices:
        Indices into ``points`` of the points that lie on the hull.
    simplices:
        ``(T, 3)`` triangles, counter-clockwise seen from outside.
    equations:
        ``(T, 4)`` plane equations ``n . x + d = 0`` with outward unit ``n``;
        the plane offset from the origin is ``-d``.
    neighbours:
        Mapping ``vertex index -> sorted list of adjacent hull vertices``,
        which is the form Appendix B's algorithm produces natively ("the hull
        obtained the way described above consists of the lists of vertices
        connected to each vertex").  Computed on first access and cached:
        Minkowski minimisation builds thousands of hulls inside its line search
        and needs none of them, so building this eagerly cost more than the
        hulls themselves.
    """

    points: np.ndarray
    vertices: np.ndarray
    simplices: np.ndarray
    equations: np.ndarray
    _neighbours: "dict[int, list[int]] | None" = None

    @property
    def neighbours(self) -> dict[int, list[int]]:
        """Adjacency lists, computed on first access."""
        if self._neighbours is None:
            self._neighbours = _neighbours_from_simplices(self.simplices)
        return self._neighbours

    @property
    def n_facets(self) -> int:
        """Number of triangles on the hull."""
        return int(self.simplices.shape[0])

    def vertex_points(self) -> np.ndarray:
        """The hull vertices as an ``(V, 3)`` coordinate array."""
        return self.points[self.vertices]

    def merge_coplanar(self, tol: float = 1e-9) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        """Group coplanar triangles into polygonal facets (Appendix B, step 8).

        Parameters
        ----------
        tol:
            Tolerance on the plane coefficients used to decide coplanarity.

        Returns
        -------
        polygons:
            One list of point indices per merged facet, ordered
            counter-clockwise about the outward normal.
        normals:
            ``(P, 3)`` outward unit normals.
        offsets:
            ``(P,)`` distances of the facet planes from the origin.
        """
        eq = np.asarray(self.equations, dtype=float)
        # Greedy clustering on the plane coefficients; a fixed rounding grid
        # would split planes that straddle a bucket boundary.
        reps: list[np.ndarray] = []
        labels = np.empty(eq.shape[0], dtype=np.int64)
        for t, plane in enumerate(eq):
            if reps:
                diff = np.abs(np.asarray(reps) - plane).max(axis=1)
                hit = int(np.argmin(diff))
                if diff[hit] <= tol:
                    labels[t] = hit
                    continue
            labels[t] = len(reps)
            reps.append(plane)

        polygons: list[list[int]] = []
        normals: list[np.ndarray] = []
        offsets: list[float] = []
        for group in range(len(reps)):
            sel = labels == group
            idx = np.unique(self.simplices[sel].ravel())
            n = eq[sel][0, :3]
            n = n / np.linalg.norm(n)
            order = _order_polygon(self.points[idx], n)
            polygons.append([int(idx[i]) for i in order])
            normals.append(n)
            offsets.append(-float(np.mean(eq[sel][:, 3])))
        return polygons, np.asarray(normals), np.asarray(offsets)


def _order_polygon(pts: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Boundary of coplanar ``pts``, counter-clockwise about ``normal``.

    Step 8 of Appendix B also asks to "remove those points that do not define
    new planes", so points strictly inside the planar face are dropped: they
    lie on the hull surface and contribute neither area nor volume.  The
    in-plane boundary is found with a monotone-chain 2-D convex hull.
    """
    if len(pts) < 3:  # pragma: no cover - degenerate face
        return np.arange(len(pts))
    centre = pts.mean(axis=0)
    rel = pts - centre
    # Any orthonormal in-plane basis (u, v) with u x v along `normal`.
    seed = rel[int(np.argmax(np.linalg.norm(rel, axis=1)))]
    u = seed - (seed @ normal) * normal
    nrm = np.linalg.norm(u)
    if nrm < 1e-15:  # pragma: no cover - all points coincide
        return np.arange(len(pts))
    u = u / nrm
    v = np.cross(normal, u)
    return _hull2d(np.column_stack([rel @ u, rel @ v]))


def _hull2d(xy: np.ndarray) -> np.ndarray:
    """Counter-clockwise 2-D convex hull indices (Andrew's monotone chain)."""
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    scale = max(1.0, float(np.abs(xy).max()))
    eps = 1e-12 * scale * scale

    def _build(seq: np.ndarray) -> list[int]:
        chain: list[int] = []
        for i in seq:
            while len(chain) >= 2:
                o, a, b = xy[chain[-2]], xy[chain[-1]], xy[i]
                cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
                if cross > eps:
                    break
                chain.pop()
            chain.append(int(i))
        return chain

    lower, upper = _build(order), _build(order[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.int64)


def convex_hull(points: np.ndarray, method: str = "auto") -> HullResult:
    """Convex hull of a point cloud.

    Parameters
    ----------
    points:
        ``(N, 3)`` array.
    method:
        ``"qhull"`` (via SciPy), ``"giftwrap"`` for the literal Appendix B
        algorithm, or ``"auto"`` (default).  Both produce identical geometry.
        Appendix B recommends its own ``N**2`` method for ``N`` "less than,
        say, 1000", noting that faster library routines exist but that
        "``N**2`` is foolproof"; ``"auto"`` follows that advice and hands
        larger clouds to Qhull.

    Returns
    -------
    HullResult
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if method == "auto":
        method = "giftwrap" if len(pts) < 1000 else "qhull"
    if method == "giftwrap":
        return gift_wrap_hull(pts)
    if method != "qhull":
        raise ValueError(f"unknown method {method!r}; use 'qhull', 'giftwrap' or 'auto'")

    try:
        # Q12 tolerates the wide merges that near-degenerate clouds provoke;
        # anything worse is reported as a ConvexHullError.
        hull = _QhullConvexHull(pts, qhull_options="Qt Q12")
    except _QhullError as exc:
        raise ConvexHullError(f"Qhull failed on {len(pts)} points: {exc}") from exc
    simplices = _orient_outward(pts, np.asarray(hull.simplices), np.asarray(hull.equations))
    return HullResult(
        points=pts,
        vertices=np.asarray(hull.vertices),
        simplices=simplices,
        equations=np.asarray(hull.equations, dtype=float),
    )


def _orient_outward(pts: np.ndarray, simplices: np.ndarray, equations: np.ndarray) -> np.ndarray:
    """Flip triangle winding so that ``(b-a) x (c-a)`` matches the outward normal."""
    a, b, c = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    cross = np.cross(b - a, c - a)
    flip = np.einsum("ij,ij->i", cross, equations[:, :3]) < 0
    out = simplices.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def _neighbours_from_simplices(simplices: np.ndarray) -> dict[int, list[int]]:
    """Vertex adjacency of a triangulated hull, as sorted lists."""
    tri = np.asarray(simplices, dtype=np.int64)
    if tri.size == 0:  # pragma: no cover - empty hull
        return {}
    a = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    b = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    # Both directions, then deduplicate as pairs.
    u = np.concatenate([a, b])
    v = np.concatenate([b, a])
    pairs = np.unique(np.column_stack([u, v]), axis=0)
    keys, starts = np.unique(pairs[:, 0], return_index=True)
    ends = np.append(starts[1:], len(pairs))
    return {
        int(k): pairs[s:e, 1].tolist() for k, s, e in zip(keys, starts, ends, strict=True)
    }


# --------------------------------------------------------------------------
# Appendix B: gift wrapping
# --------------------------------------------------------------------------
def gift_wrap_hull(points: np.ndarray, tol: float = 1e-12) -> HullResult:
    """Convex hull by gift wrapping, following Appendix B of the paper.

    The appendix rotates a plane about each hull vertex in turn; the
    equivalent and slightly tidier bookkeeping used here rotates the plane
    about each hull *edge*, which produces the facets directly.  The pivot
    test is the same sign test the paper writes as
    ``r_ad . (r_ac x r_ab) > 0``.

    Parameters
    ----------
    points:
        ``(N, 3)`` array; duplicate points are tolerated.
    tol:
        Degeneracy tolerance.

    Returns
    -------
    HullResult
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 4:
        raise ValueError("need at least 4 points for a 3-D hull")

    first = _initial_facet(pts, tol)
    simplices: list[tuple[int, int, int]] = [first]
    # Directed edges that already belong to an emitted facet.  The facet on
    # the far side of edge (x, y) traverses it as (y, x), so that reversed
    # edge is what still needs wrapping.
    done: set[tuple[int, int]] = set()
    queue: list[tuple[int, int, np.ndarray]] = []

    def _register(tri: tuple[int, int, int]) -> None:
        n = _facet_normal(pts, *tri)
        for i in range(3):
            x, y = tri[i], tri[(i + 1) % 3]
            done.add((x, y))
            if (y, x) not in done:
                queue.append((x, y, n))

    _register(first)
    while queue:
        u, v, n_prev = queue.pop()
        if (v, u) in done:
            continue
        w = _pivot(pts, u, v, n_prev, tol)
        if w is None:  # pragma: no cover - only for degenerate inputs
            continue
        tri = (v, u, w)
        simplices.append(tri)
        _register(tri)

    simplices_arr = _retriangulate_flat_faces(pts, np.asarray(simplices, dtype=np.int64))
    normals = np.array([_facet_normal(pts, *tri) for tri in simplices_arr])
    if simplices_arr.ndim != 2 or len(simplices_arr) < 4:
        raise ConvexHullError(
            f"gift wrapping produced {len(simplices_arr)} facets from "
            f"{len(pts)} points; the cloud is degenerate (collapsed, "
            "collinear or coplanar)"
        )
    offsets = -np.einsum("ij,ij->i", normals, pts[simplices_arr[:, 0]])
    return HullResult(
        points=pts,
        vertices=np.unique(simplices_arr.ravel()),
        simplices=simplices_arr,
        equations=np.column_stack([normals, offsets]),
    )


def _retriangulate_flat_faces(pts: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Rebuild the triangulation face by face (Appendix B, step 8).

    Where more than three hull points share a plane the edge pivot has a tie,
    and resolving each tie independently can tile a flat face inconsistently.
    Grouping the triangles by plane, taking each face's in-plane boundary and
    fanning it out removes the ambiguity - which is exactly what step 8 of the
    appendix prescribes.
    """
    normals = np.array([_facet_normal(pts, *tri) for tri in simplices])
    offsets = -np.einsum("ij,ij->i", normals, pts[simplices[:, 0]])
    provisional = HullResult(
        points=pts,
        vertices=np.unique(simplices.ravel()),
        simplices=simplices,
        equations=np.column_stack([normals, offsets]),
    )
    scale = max(1.0, float(np.abs(pts).max()))
    polygons, _, _ = provisional.merge_coplanar(tol=1e-9 * scale)
    tris: list[tuple[int, int, int]] = []
    for poly in polygons:
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
    return np.asarray(tris, dtype=np.int64)


def _facet_normal(pts: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    n = np.cross(pts[j] - pts[i], pts[k] - pts[i])
    nrm = np.linalg.norm(n)
    return n / nrm if nrm > 0 else n


def _initial_facet(pts: np.ndarray, tol: float) -> tuple[int, int, int]:
    """Appendix B steps 1-2: the first hull facet.

    ``a`` is the point with the smallest ``z`` (step 1: "the point that has
    the largest/smallest value of x, y, or z").  ``b`` is then the point whose
    direction from ``a`` is the most extreme - here, the smallest elevation
    above the horizontal plane through ``a`` - which guarantees that ``a-b``
    is a hull edge (step 2).  A downward-facing plane through that edge is the
    pivot's starting orientation; one pivot step completes the facet, whose
    winding is then fixed so that its normal points outward.
    """
    a = int(np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2]))[0])
    down = np.array([0.0, 0.0, -1.0])
    # Rotate the horizontal supporting plane at `a` about a fixed horizontal
    # line through `a`; the first point it meets is a hull neighbour of `a`
    # and the rotated plane supports the hull along the edge a-b.
    axis = np.array([1.0, 0.0, 0.0])
    w = np.cross(axis, down)
    rel = pts - pts[a]
    proj_down, proj_w = rel @ down, rel @ w
    phi = np.arctan2(-proj_down, proj_w)
    phi = np.where(phi < -tol, phi + np.pi, phi)  # the plane is two-sided
    phi[a] = np.inf
    phi[np.linalg.norm(rel, axis=1) <= tol] = np.inf
    if not np.isfinite(phi).any():  # pragma: no cover - all points coincide
        raise ConvexHullError("degenerate point set: no two distinct points")
    b = int(np.argmin(phi))
    n0 = np.cos(phi[b]) * down + np.sin(phi[b]) * w

    c = _pivot(pts, a, b, n0, tol)
    if c is None:  # pragma: no cover - collinear point set
        raise ConvexHullError("degenerate point set: all points are collinear")

    tri = (b, a, c)
    n = _facet_normal(pts, *tri)
    # Every point must lie on the inner side of a hull facet plane.
    if float(((pts - pts[a]) @ n).max()) > tol * max(1.0, float(np.abs(pts).max())):
        tri = (a, b, c)
    return tri


def _pivot(pts: np.ndarray, u: int, v: int, n_prev: np.ndarray, tol: float) -> int | None:
    """Rotate the plane about the directed edge ``u -> v`` and return the hit point.

    The candidate minimising the rotation angle is the next hull vertex; this
    is the vectorised form of Appendix B step 3, where each candidate ``d`` is
    accepted when it lies above the plane spanned by the current base line and
    the running reference point ``c``.
    """
    edge = pts[v] - pts[u]
    nrm = np.linalg.norm(edge)
    if nrm < tol:  # pragma: no cover
        return None
    e = edge / nrm
    # In-plane direction pointing away from the current facet across the edge.
    t = np.cross(e, n_prev)

    rel = pts - pts[u]
    perp = rel - np.outer(rel @ e, e)
    lengths = np.linalg.norm(perp, axis=1)
    # `n_prev` supports the hull, so no point sits above it and the exact
    # rotation angle lies in [0, pi].  Points *in* the plane of the current
    # facet evaluate to a signed zero, and a -1e-17 there would make arctan2
    # return -pi instead of +pi and hand the pivot back a vertex of the facet
    # it came from; snap that height to zero before taking the angle.
    height = perp @ (-n_prev)
    height[np.abs(height) <= 1e-9 * np.maximum(lengths, tol)] = 0.0
    ang = np.arctan2(height, perp @ t)
    ang = np.where(ang < 0.0, ang + 2.0 * np.pi, ang)
    ang[lengths <= tol] = np.inf
    ang[[u, v]] = np.inf
    if not np.isfinite(ang).any():  # pragma: no cover
        return None
    # Break ties among coplanar candidates by taking the farthest one, which
    # keeps the emitted triangles non-degenerate on flat faces.
    best_ang = float(np.min(ang))
    tied = np.flatnonzero(ang <= best_ang + 1e-9)
    return int(tied[np.argmax(lengths[tied])])


def hull_volume_area(hull: HullResult) -> tuple[float, float]:
    """Volume and total surface area of a :class:`HullResult`."""
    p = hull.points
    a, b, c = p[hull.simplices[:, 0]], p[hull.simplices[:, 1]], p[hull.simplices[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * float(np.linalg.norm(cross, axis=1).sum())
    volume = float(np.einsum("ij,ij->i", a, cross).sum() / 6.0)
    return volume, area
