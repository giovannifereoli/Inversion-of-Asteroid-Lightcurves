"""Appendix C - Minkowski minimisation.

    As explained in KLLB and Lamberg (1993), the reconstruction of the convex
    polyhedron corresponding to given facet areas g and surface normals can be
    expressed as a constrained minimization problem where l, the distances of
    the facet planes from the origin, are to be solved for.  The object
    function is the inner product <l, g> in R^n-space, while the constraint
    function is V(l), the volume of the polyhedron computed from l.  In
    practice, the equivalent procedure of maximizing V(l) while staying on the
    hyperplane <l, g> = constant is computationally more efficient as the
    constraint function is now linear.

Inversion (Section 3.1) recovers only the *areas* of the facets; this module
is what turns them back into a body: "once the areas of the facets are known,
the vertices of the facets can be obtained by Minkowski minimization".

The volume is Eq. (17), ``V = (1/3) sum_j l_j A_j(l)``, its gradient is ``A``,
and the projection onto the constraint plane is Eq. (18),
``f = A - (<A, g> / <g, g>) g``.  ``A_j(l)`` itself is evaluated through the
dual transform Eq. (19), ``r = n / l``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .convexhull import convex_hull, hull_volume_area
from .mesh import Polyhedron

__all__ = [
    "MinkowskiResult",
    "minkowski_solve",
    "close_facet_areas",
    "merge_duplicate_normals",
    "dual_polyhedron",
]


@dataclass
class MinkowskiResult:
    """Outcome of :func:`minkowski_solve`.

    Attributes
    ----------
    polyhedron:
        The reconstructed convex body, scaled so its facet areas match ``g``.
    distances:
        ``(M,)`` plane distances ``l`` from the origin.
    areas:
        ``(M,)`` realised facet areas, aligned with the *input* normals; zero
        where a normal does not appear on the body.
    dark_area:
        Area of the extra closing facet of Section 3.3, or ``0`` if none was
        needed.  The paper notes it "is completely dark, but its existence does
        not affect the overall shape".
    n_iterations:
        Conjugate-gradient iterations actually used.
    alignment:
        ``<A, g> / (|A| |g|)``: 1 when the realised areas are exactly
        proportional to the requested ones, which is the optimum of the
        constrained problem.
    converged:
        Whether the solve finished for a good reason - either ``alignment``
        reached ``tol``, or the *body* stopped changing (see ``status``).
    status:
        ``"converged"`` (alignment tolerance met), ``"shape_converged"`` (the
        axis ratios stopped moving, which is what matters physically),
        ``"stalled"`` (conjugate gradients could make no further progress) or
        ``"max_iter"``.
    shape_change:
        Largest relative change in the body's principal extents over the last
        iteration.  This, not ``alignment``, is the number to judge a solution
        by: ``V(l)`` is very ill-conditioned near its maximum - hundreds of
        near-zero facets contribute almost nothing - so ``alignment`` keeps
        creeping long after the shape has settled.
    volume:
        Volume of the returned body.
    """

    polyhedron: Polyhedron
    distances: np.ndarray
    areas: np.ndarray
    dark_area: float
    n_iterations: int
    alignment: float
    converged: bool
    status: str
    shape_change: float
    volume: float


def merge_duplicate_normals(
    normals: np.ndarray, areas: np.ndarray, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine facets that share a normal direction into one.

    A triangulated convex hull represents each large planar face as several
    coplanar triangles with identical normals - and Section 3.5 stresses that
    such faces are the interesting part, the "large planar sections" that mark
    concavities.  The dual transform Eq. (19) maps identical normals to
    identical points, which is degenerate, so duplicates are summed here into
    the single facet they physically are.

    Parameters
    ----------
    normals:
        ``(M, 3)`` unit normals.
    areas:
        ``(M,)`` facet areas.
    tol:
        Angular tolerance for treating two normals as equal.

    Returns
    -------
    normals, areas:
        The reduced set, with areas summed over each group.
    inverse:
        ``(M,)`` index of each input facet in the reduced set.
    """
    n = np.asarray(normals, dtype=float)
    g = np.asarray(areas, dtype=float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    _, first, inverse = np.unique(
        np.round(n / tol).astype(np.int64), axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.ravel()
    merged = np.zeros(len(first))
    np.add.at(merged, inverse, g)
    return n[first], merged, inverse


def close_facet_areas(
    normals: np.ndarray, areas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Append one dark facet so that Eq. (3) holds exactly.

    Section 3.3: "A small nonzero residual in (3) is easily fixed by adding a
    facet of corresponding size such that the new (3) vanishes: this is done to
    make sure that Minkowski minimization proceeds successfully.  The small new
    facet is completely dark, but its existence does not affect the overall
    shape."

    Parameters
    ----------
    normals:
        ``(M, 3)`` unit normals ``n_j``.
    areas:
        ``(M,)`` facet areas ``g_j``.

    Returns
    -------
    normals, areas:
        Copies with one extra entry, unless the residual is already negligible.
    """
    n = np.asarray(normals, dtype=float)
    g = np.asarray(areas, dtype=float)
    residual = g @ n
    size = float(np.linalg.norm(residual))
    if size <= 1e-12 * float(g.sum()):
        return n, g
    extra = -residual / size
    return np.vstack([n, extra]), np.concatenate([g, [size]])


def _home_vertices(normals: np.ndarray, l: np.ndarray, tol: float) -> np.ndarray:
    """Vertices of ``{x : n_j . x <= l_j}``, via the Eq. (19) dual transform."""
    hull = convex_hull(normals / l[:, None], method="qhull")
    offset = -hull.equations[:, 3]
    if np.any(offset <= 0):  # pragma: no cover - origin outside the dual hull
        raise ValueError("dual hull does not contain the origin")
    home = hull.equations[:, :3] / offset[:, None]
    # Coplanar dual facets (a home vertex where more than three facets meet)
    # are triangulated by Qhull and map to the same home point.
    scale = max(float(np.abs(home).max()), 1e-12)
    _, keep = np.unique(np.round(home / (tol * scale)).astype(np.int64), axis=0,
                        return_index=True)
    return home[keep]


def dual_polyhedron(
    normals: np.ndarray, distances: np.ndarray, tol: float = 1e-9
) -> tuple[Polyhedron, np.ndarray]:
    """Build the body ``{x : n_j . x <= l_j}`` through the dual transform.

    Appendix C:

        Constructing the polyhedron from l is easiest via the so-called dual
        transform.  This transform maps a plane with the surface unit normal n
        and distance l from the origin into a point given by the radius vector
        ``r = n / l``, and vice versa. [...] The important point is that
        adjacency information is retained; i.e., the vertices of a facet become
        the facets surrounding a vertex.

    Because the transform is an involution, the hull of the dual points has one
    facet per home *vertex*; dualising those facets back gives the home
    vertices, and the home body is simply their convex hull.  Each of its
    triangles is then attributed to the requested normal it is parallel to,
    which recovers ``A_j``.

    Parameters
    ----------
    normals:
        ``(M, 3)`` unit normals.
    distances:
        ``(M,)`` strictly positive plane distances.
    tol:
        Relative tolerance for merging coincident home vertices.

    Returns
    -------
    body:
        The polyhedron, triangulated.
    areas:
        ``(M,)`` area of each requested normal's facet; zero when that normal
        does not appear on the body.
    """
    n = np.ascontiguousarray(normals, dtype=float)
    l = np.ascontiguousarray(distances, dtype=float)
    if np.any(l <= 0):
        raise ValueError("all plane distances must be positive")

    vertices = _home_vertices(n, l, tol)
    hull = convex_hull(vertices, method="qhull")
    used, inverse = np.unique(hull.simplices, return_inverse=True)
    body = Polyhedron(vertices[used], inverse.reshape(-1, 3), validate=False)

    # Attribute every triangle to the plane it lies in.  Coplanar triangles of
    # one home facet share a normal, so this reassembles A_j exactly.
    owner = np.argmax(body.normals @ n.T, axis=1)
    areas = np.zeros(len(n))
    np.add.at(areas, owner, body.areas)
    return body, areas


def _volume(normals: np.ndarray, l: np.ndarray, tol: float) -> float:
    """Eq. (17) only - the cheap path used inside the line search."""
    vertices = _home_vertices(normals, l, tol)
    return float(hull_volume_area(convex_hull(vertices, method="qhull"))[0])


def _centroid(normals: np.ndarray, l: np.ndarray, tol: float) -> np.ndarray:
    """Centre of mass of the body, for Appendix C's per-iteration recentring."""
    vertices = _home_vertices(normals, l, tol)
    hull = convex_hull(vertices, method="qhull")
    return Polyhedron(vertices, hull.simplices, validate=False).centroid


def minkowski_solve(
    normals: np.ndarray,
    areas: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-7,
    shape_tol: float = 1e-5,
    patience: int = 30,
    close: bool = True,
    center: bool = True,
    verbose: bool = False,
) -> MinkowskiResult:
    """Recover the convex body whose facet areas are ``areas``.

    Maximises Eq. (17) along the projected gradient of Eq. (18) using
    Polak-Ribiere conjugate gradients, exactly as Appendix C prescribes: "the
    direction and size of the iteration step can be determined using standard
    methods such as conjugate gradients; when implementing line minimization,
    any trial steps [...] leading to negative values for any l_i must be
    contracted back to the positive region by, e.g., bisection".

    Parameters
    ----------
    normals:
        ``(M, 3)`` unit facet normals ``n_j``.
    areas:
        ``(M,)`` positive facet areas ``g_j``, e.g. from
        :class:`~lcinv.convex.ConvexInversion`.
    max_iter:
        Maximum conjugate-gradient iterations.
    tol:
        Convergence threshold on ``1 - alignment``.  The default of ``1e-7``
        already pins the axis ratios to well below a part in a thousand.
    shape_tol:
        Alternative, physically meaningful stopping rule: finish once the
        body's principal extents change by less than this fraction for three
        consecutive iterations (or when a stall leaves them that stable).
        ``1e-5`` is a change of one part in 100,000 in the axis ratios - far
        below anything photometry can constrain.  ``V(l)`` is ill-conditioned enough that
        ``alignment`` can grind for hundreds of iterations after the shape has
        stopped moving, so this is usually what ends the solve.
    patience:
        Give up after this many consecutive iterations that fail to improve on
        the best ``1 - alignment`` seen so far.  Reported as
        ``status="stalled"``.
    close:
        Apply :func:`close_facet_areas` first, per Section 3.3.
    center:
        Shift the centroid to the origin each iteration.  Appendix C: "it is
        useful to shift the centroid of the polyhedron to the origin at each
        iteration step; i.e., the elements l_i of the present vector l are
        changed to ``l_i - n_i . r_c``".  This is a pure gauge move - it leaves
        the body unchanged, and leaves ``<l, g>`` unchanged too whenever Eq.
        (3) holds.
    verbose:
        Print the alignment at each iteration.

    Returns
    -------
    MinkowskiResult

    Notes
    -----
    Appendix C ends by scaling "each vertex coordinate [...] with the factor
    ``sqrt(|A|/|g|)``".  The constraint fixes ``<l, g>`` but not the scale of
    the *areas*, so at the optimum ``A = c g`` and one needs ``c s^2 = 1``,
    i.e. ``s = sqrt(|g| / |A|)`` - the reciprocal of the printed expression.
    This implementation uses the equivalent, better-conditioned
    ``s = sqrt(<g, g> / <A, g>)``, which reproduces the unit sphere exactly
    from its own facet areas (see the tests).
    """
    n = np.ascontiguousarray(normals, dtype=float)
    g = np.ascontiguousarray(areas, dtype=float).ravel()
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError("normals must have shape (M, 3)")
    if len(g) != len(n):
        raise ValueError("normals and areas must have the same length")
    if np.any(g < 0):
        raise ValueError("facet areas must be non-negative")
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    requested = g.copy()
    n, g, inverse = merge_duplicate_normals(n, g)
    merged = g.copy()
    n_merged = len(g)
    if close:
        n, g = close_facet_areas(n, g)

    gg = float(g @ g)
    # Appendix C's suggested start: "set each l_j to <g, g> / sum g_j".
    l = np.full(len(g), gg / float(g.sum()))
    target = float(l @ g)
    geo_tol = 1e-9

    direction = np.zeros_like(l)
    prev_f = None
    alignment = 0.0
    it = 0
    status = "max_iter"
    shape_change = np.inf
    prev_extents: np.ndarray | None = None
    stable = 0
    best_residual = np.inf
    since_best = 0
    body: Polyhedron | None = None
    realised = np.zeros_like(g)

    for it in range(1, max_iter + 1):
        body, realised = dual_polyhedron(n, l, geo_tol)
        f = realised - (realised @ g) / gg * g  # Eq. (18)

        norm_a = float(np.linalg.norm(realised))
        alignment = float(realised @ g) / max(norm_a * np.sqrt(gg), 1e-300)
        residual = 1.0 - alignment
        extents = body.extents()
        extents = extents / max(extents[-1], 1e-300)
        if prev_extents is not None:
            shape_change = float(np.abs(extents - prev_extents).max())
            stable = stable + 1 if shape_change < shape_tol else 0
        prev_extents = extents
        if verbose:  # pragma: no cover - diagnostic only
            print(
                f"  minkowski {it:3d}  1-alignment={residual:.3e}"
                f"  shape change={shape_change:.2e}"
            )
        if residual < tol:
            status = "converged"
            break
        if stable >= 3:
            # The axis ratios have stopped moving; further alignment gains do
            # not change the body.
            status = "shape_converged"
            break
        # Conjugate gradients progresses in plateaus here, so "stalled" must
        # mean no new best at all for a long stretch - not merely slow.
        if residual < best_residual:
            best_residual, since_best = residual, 0
        else:
            since_best += 1
            if since_best >= patience:
                # A stall with a settled body is still a good answer: report it
                # as such rather than as a failure.
                status = "shape_converged" if shape_change < shape_tol else "stalled"
                break

        # Polak-Ribiere, restarted whenever it stops being an ascent direction.
        if prev_f is None:
            direction = f.copy()
        else:
            beta = max(0.0, float(f @ (f - prev_f)) / max(float(prev_f @ prev_f), 1e-300))
            direction = f + beta * direction
            if float(direction @ f) <= 0.0:
                direction = f.copy()
        prev_f = f.copy()

        # Largest step that keeps every l_j positive, then bounded line search.
        neg = direction < 0
        t_max = 0.99 * float(np.min(-l[neg] / direction[neg])) if neg.any() else 1.0
        if not np.isfinite(t_max) or t_max <= 0:  # pragma: no cover
            break
        if not neg.any():
            t_max = float(np.linalg.norm(l) / max(np.linalg.norm(direction), 1e-300))

        def negative_volume(t: float, l=l, direction=direction) -> float:
            # `l` and `direction` are bound as defaults: the closure is only
            # ever called inside this iteration, but binding makes that explicit.
            trial = l + t * direction
            if np.any(trial <= 0):  # pragma: no cover - guarded by t_max
                return np.inf
            try:
                return -_volume(n, trial, geo_tol)
            except ValueError:  # pragma: no cover - degenerate trial geometry
                return np.inf

        step = minimize_scalar(
            negative_volume, bounds=(0.0, t_max), method="bounded",
            # Conjugate gradients on this problem is sensitive to the accuracy
            # of the line minimum, so the search is kept tight.
            options={"xatol": 1e-8 * t_max, "maxiter": 24},
        )
        if not np.isfinite(step.fun):  # pragma: no cover
            break
        l = l + float(step.x) * direction

        if center:
            l = l - n @ _centroid(n, l, geo_tol)
        # Restore the constraint exactly; centring alone preserves it only when
        # Eq. (3) holds for g.
        l = l + (target - float(l @ g)) / gg * g
        if np.any(l <= 0):  # pragma: no cover - recover from a bad projection
            l = np.maximum(l, 1e-6 * float(np.abs(l).max()))

    body, realised = dual_polyhedron(n, l, geo_tol)
    scale = float(np.sqrt(gg / max(float(realised @ g), 1e-300)))
    body = body.scaled(scale)
    if center:
        body = body.centered()

    # Undo the merge so the caller sees one area per normal it passed in,
    # splitting each merged facet in the ratio of the areas that were asked for.
    scaled = realised * scale**2
    share = np.divide(
        requested, merged[inverse], out=np.ones_like(requested), where=merged[inverse] > 0
    )
    out_areas = scaled[:n_merged][inverse] * share
    dark = float(scaled[n_merged]) if len(scaled) > n_merged else 0.0

    return MinkowskiResult(
        polyhedron=body,
        distances=l * scale,
        areas=out_areas,
        dark_area=dark,
        n_iterations=it,
        alignment=alignment,
        converged=status in ("converged", "shape_converged"),
        status=status,
        shape_change=float(shape_change),
        volume=body.volume,
    )
