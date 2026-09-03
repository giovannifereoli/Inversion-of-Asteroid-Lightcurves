"""Figures.

The paper's own figures set the pattern and this module reproduces it:

* Figures 1-4 show, for each test body, "the original shape shown from two
  directions", its convex hull, the inversion result, and the lightcurves the
  three of them produce - :func:`plot_shape_views` and
  :func:`plot_lightcurve_comparison`;
* Figure 5 shows a nonconvex solution "seen and illuminated from two
  directions" - the same shape plotting.

Matplotlib is an optional dependency; import this module only when plotting.
"""

from __future__ import annotations

import numpy as np

from .lightcurve import LightcurveSet
from .mesh import Polyhedron
from .scattering import LommelSeeligerLambert

__all__ = [
    "plot_shape_views",
    "plot_lightcurve_comparison",
    "plot_lightcurve_grid",
    "plot_period_scan",
    "plot_pole_samples",
    "plot_corner",
    "plot_facet_values",
]

_VIEW_LABELS = ("+x (equator)", "+y (equator)", "+z (pole)")
_VIEWS = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def _shade(body: Polyhedron, view: np.ndarray, sun: np.ndarray | None = None) -> tuple:
    """Painter's-algorithm polygons for one orthographic view."""
    view = view / np.linalg.norm(view)
    sun = view if sun is None else sun / np.linalg.norm(sun)

    # An in-plane basis for the projection.
    helper = np.array([0.0, 0.0, 1.0]) if abs(view[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)

    normals = body.normals
    facing = normals @ view > 0
    tri = body.vertices[body.facets[facing]]
    depth = (tri.mean(axis=1) @ view)
    order = np.argsort(depth)
    tri = tri[order]

    polys = np.stack([tri @ right, tri @ up], axis=-1)
    lit = np.clip(normals[facing][order] @ sun, 0.0, 1.0)
    return polys, lit


def plot_shape_views(
    body: Polyhedron,
    axes=None,
    title: str = "",
    colour: str = "#c9b8a0",
    sun: np.ndarray | None = None,
):
    """Three orthographic views of a body, as DAMIT renders its models.

    Parameters
    ----------
    body:
        The polyhedron to draw.
    axes:
        Three matplotlib axes; created if omitted.
    title:
        Figure title.
    colour:
        Base facet colour.
    sun:
        Illumination direction; the view direction is used when omitted, which
        gives the flat "artificial light" look.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    created = axes is None
    if created:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    else:
        fig = np.atleast_1d(axes)[0].figure

    span = float(np.abs(body.vertices).max()) * 1.08
    base = np.array(plt.matplotlib.colors.to_rgb(colour))
    for ax, view, label in zip(np.atleast_1d(axes), _VIEWS, _VIEW_LABELS, strict=True):
        polys, lit = _shade(body, view, sun)
        shades = base[None, :] * (0.25 + 0.75 * lit)[:, None]
        ax.add_collection(
            PolyCollection(polys, facecolors=shades, edgecolors="none", antialiaseds=True)
        )
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    if created:
        fig.tight_layout()
    return fig


def plot_lightcurve_comparison(
    bodies: dict[str, Polyhedron],
    geometries: list[tuple[np.ndarray, np.ndarray]],
    law=None,
    n_points: int = 120,
    axes=None,
    rescale: bool = True,
):
    """Reproduce panel (d) of Figures 1-4: one body's curve against another's.

    Parameters
    ----------
    bodies:
        Label to body.  The first entry is the reference for Eq. (14) rescaling.
    geometries:
        One ``(earth, sun)`` pair of body-frame unit vectors per panel; the
        body is rotated about ``z`` through a full turn for each.
    law:
        Scattering law; the paper's equal-weight Lommel-Seeliger + Lambert
        by default.
    n_points:
        Points per synthetic curve.
    axes:
        Matplotlib axes, one per geometry.
    rescale:
        Apply Eq. (14) so that the curves are compared shape-to-shape rather
        than size-to-size, as Section 3.5 does for the convex hull.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from .lightcurve import optimal_scale
    from .raytracer import RayTracer

    law = law or LommelSeeligerLambert(1.0)
    created = axes is None
    if created:
        fig, axes = plt.subplots(1, len(geometries), figsize=(5.2 * len(geometries), 3.6))
    else:
        fig = np.atleast_1d(axes)[0].figure
    axes = np.atleast_1d(axes)

    phase = np.linspace(0.0, 2.0 * np.pi, n_points)
    styles = ["-", ":", "--", "-."]
    tracers = {k: RayTracer(v) for k, v in bodies.items()}

    curves: dict[str, list[np.ndarray]] = {k: [] for k in bodies}
    for earth, sun in geometries:
        for name, tracer in tracers.items():
            rot = np.stack(
                [
                    [np.cos(phase), -np.sin(phase), np.zeros_like(phase)],
                    [np.sin(phase), np.cos(phase), np.zeros_like(phase)],
                    [np.zeros_like(phase), np.zeros_like(phase), np.ones_like(phase)],
                ]
            )
            e = np.einsum("ijk,j->ki", rot, np.asarray(earth, dtype=float))
            s = np.einsum("ijk,j->ki", rot, np.asarray(sun, dtype=float))
            curves[name].append(tracer.lightcurve(e, s, law))

    reference = next(iter(bodies))
    scales = {reference: 1.0}
    if rescale:
        ref_all = np.concatenate(curves[reference])
        for name in bodies:
            if name != reference:
                scales[name] = optimal_scale(ref_all, np.concatenate(curves[name]))

    for k, ax in enumerate(axes):
        for name, style in zip(bodies, styles, strict=False):
            ax.plot(
                phase / (2 * np.pi), curves[name][k] * scales[name],
                style, lw=1.6, label=name,
            )
        ax.set_xlabel("rotational phase")
        ax.set_ylabel("brightness")
        ax.legend(fontsize=8, frameon=False)
    if created:
        fig.tight_layout()
    return fig


def plot_lightcurve_grid(
    data: LightcurveSet,
    models: list[np.ndarray] | None = None,
    max_curves: int = 12,
    n_cols: int = 4,
    normalise: bool = True,
):
    """Observed lightcurves with the model overlaid, one panel per curve.

    Parameters
    ----------
    data:
        The observations.
    models:
        Model brightnesses per curve, as returned in
        :attr:`~lcinv.convex.InversionResult.model_lightcurves`.
    max_curves:
        Draw at most this many, chosen to span the observing geometries.
    n_cols:
        Panels per row.
    normalise:
        Plot ``L / L-bar``, the quantity Eq. (13) actually fits.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    index = list(range(len(data)))
    if len(index) > max_curves:
        step = len(index) / max_curves
        index = [int(i * step) for i in range(max_curves)]

    n_rows = int(np.ceil(len(index) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, index, strict=False):
        curve = data[i]
        y = curve.normalised if normalise else curve.brightness
        t = curve.jd - curve.jd.min()
        ax.plot(t, y, "o", ms=2.6, color="#3b6ea5", label="observed")
        if models is not None:
            ax.plot(t, models[i], "-", lw=1.4, color="#c2410c", label="model")
        ax.set_title(
            f"{curve.name}  $\\alpha$={np.degrees(curve.mean_phase_angle):.0f}$^\\circ$",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
    for ax in axes[len(index):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.supxlabel("JD - start of curve", fontsize=9)
    fig.supylabel("relative brightness", fontsize=9)
    fig.tight_layout()
    return fig


def plot_period_scan(periods: np.ndarray, chi2: np.ndarray, best: float | None = None):
    """The ``chi^2(P)`` curve of step 1 of the Paper II recipe."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(periods, chi2, "-", lw=0.9, color="#3b6ea5")
    if best is not None:
        ax.axvline(best, color="#c2410c", lw=1.2, ls="--", label=f"best {best:.6f} h")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("period (hours)")
    ax.set_ylabel(r"$\chi^2$")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_pole_samples(lam: np.ndarray, beta: np.ndarray, truth: tuple[float, float] | None = None):
    """Posterior pole directions on an equal-area (Hammer) projection."""
    import matplotlib.pyplot as plt

    x = np.radians(np.asarray(lam) % 360.0) - np.pi
    y = np.radians(np.asarray(beta))
    fig, ax = plt.subplots(figsize=(7, 3.8), subplot_kw={"projection": "hammer"})
    ax.scatter(x, y, s=3, alpha=0.25, color="#3b6ea5", edgecolors="none")
    if truth is not None:
        ax.scatter(
            [np.radians(truth[0] % 360.0) - np.pi], [np.radians(truth[1])],
            s=90, marker="*", color="#c2410c", label="reference", zorder=5,
        )
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_title("spin-axis posterior (ecliptic)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_corner(result, parameters: list[str] | None = None, truths=None):
    """Corner plot of selected posterior parameters.

    Parameters
    ----------
    result:
        A :class:`~lcinv.bayes.BayesResult`.
    parameters:
        Labels to include; defaults to the non-shape parameters, which are the
        ones with a physical reading.
    truths:
        Reference values, aligned with ``parameters``.
    """
    import corner

    if parameters is None:
        parameters = [
            n for n in result.labels
            if not n.startswith("a[") or n == "a[0,0]"
        ]
    cols = [result.labels.index(n) for n in parameters]
    return corner.corner(
        result.samples[:, cols], labels=parameters, truths=truths,
        show_titles=True, title_fmt=".4f", title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 9},
    )


def plot_facet_values(geometry, areas: np.ndarray, title: str = "curvature function"):
    """The solved ``g_j`` as a map over the Gaussian image sphere.

    This is the quantity the inversion actually determines; the shape follows
    from it only after Minkowski minimisation.
    """
    import matplotlib.pyplot as plt

    from .geometry import unit_to_spherical

    theta, phi = unit_to_spherical(geometry.normals)
    value = np.asarray(areas) / geometry.sphere_areas
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    sc = ax.scatter(
        np.degrees(phi), np.degrees(np.pi / 2 - theta),
        c=value, s=14, cmap="magma", edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label="$G$")
    ax.set_xlabel("normal longitude (deg)")
    ax.set_ylabel("normal latitude (deg)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
