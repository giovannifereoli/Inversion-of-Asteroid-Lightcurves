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
from .style import PALETTE, diverging_cmap, sequential_cmap, series_colors, use_style

__all__ = [
    "plot_shape_views",
    "plot_lightcurve_comparison",
    "plot_lightcurve_grid",
    "plot_period_scan",
    "plot_pole_samples",
    "plot_corner",
    "plot_facet_values",
    "plot_pole_scan",
    "plot_phase_function",
    "plot_residuals",
    "plot_regularisation_scan",
    "plot_shape_row",
]

_VIEW_LABELS = ("View from $+x$ (equator)", "View from $+y$ (equator)", "View from $+z$ (pole)")


def _tokens(mode: str = "light") -> dict:
    """Apply the package style and return its colour tokens."""
    return use_style(mode)
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
    colour: str | None = None,
    sun: np.ndarray | None = None,
    mode: str = "light",
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

    tok = _tokens(mode)
    colour = colour or tok["body"]
    created = axes is None
    if created:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    else:
        fig = np.atleast_1d(axes)[0].figure

    span = float(np.abs(body.vertices).max()) * 1.08
    base = np.array(plt.matplotlib.colors.to_rgb(colour))
    for ax, view, label in zip(np.atleast_1d(axes), _VIEWS, _VIEW_LABELS, strict=True):
        polys, lit = _shade(body, view, sun)
        shades = base[None, :] * (0.25 + 0.75 * lit)[:, None]
        # Stroke each triangle in its own fill colour: with "none" the
        # antialiased gaps between adjacent polygons let the background through
        # and the body reads as a wireframe.
        ax.add_collection(
            PolyCollection(
                polys, facecolors=shades, edgecolors=shades,
                linewidths=0.5, antialiaseds=True,
            )
        )
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9.5, color=tok["text_secondary"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    if title:
        fig.suptitle(title, color=tok["text_primary"])
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

    tok = _tokens()
    phase = np.linspace(0.0, 2.0 * np.pi, n_points)
    styles = ["-", (0, (1, 1.6)), (0, (5, 2)), (0, (4, 1.5, 1, 1.5))]
    palette = series_colors()
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
        for j, (name, style) in enumerate(zip(bodies, styles, strict=False)):
            ax.plot(
                phase / (2 * np.pi), curves[name][k] * scales[name],
                ls=style, lw=2.0, color=palette[j], label=name,
            )
        ax.set_xlabel("Rotational phase")
        ax.set_ylabel("Brightness")
        ax.legend(loc="best")
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, index, strict=False):
        curve = data[i]
        y = curve.normalised if normalise else curve.brightness
        t = curve.jd - curve.jd.min()
        ax.plot(t, y, "o", ms=3.4, color=series_colors()[0], mec=_tokens()["surface"],
                mew=0.5, label="Observed", zorder=3)
        if models is not None:
            ax.plot(t, models[i], "-", lw=1.8, color=series_colors()[1],
                    label="Model", zorder=4)
        ax.set_title(
            rf"{curve.name}   $\alpha = {np.degrees(curve.mean_phase_angle):.0f}^\circ$",
            fontsize=9, color=_tokens()["text_secondary"],
        )
        ax.tick_params(labelsize=8)
    for ax in axes[len(index):]:
        ax.set_visible(False)
    axes[0].legend(loc="best", fontsize=8)
    fig.supxlabel("Days from the start of each curve", color=_tokens()["text_secondary"])
    fig.supylabel(r"Brightness $/\ \bar{L}$", color=_tokens()["text_secondary"])
    fig.tight_layout()
    return fig


def plot_period_scan(
    periods: np.ndarray,
    chi2: np.ndarray,
    best: float | None = None,
    reference: float | None = None,
    n_mark: int = 8,
):
    """``chi^2(P)`` - step 1 of the Paper II recipe.

    Step 1 is to "determine the sampling interval of the period from the
    separation between the local ``chi^2(P)`` minima".  Those minima are the
    point of the figure, so the deepest few are marked directly rather than
    left for the reader to hunt.

    Parameters
    ----------
    periods, chi2:
        The scan, as returned by :func:`lcinv.pipeline.period_scan`.
    best:
        Period to highlight; the global minimum when omitted.
    reference:
        A published period to mark for comparison.
    n_mark:
        How many of the deepest local minima to circle.
    """
    import matplotlib.pyplot as plt

    tok = _tokens()
    periods = np.asarray(periods, dtype=float)
    chi2 = np.asarray(chi2, dtype=float)
    palette = series_colors()
    if best is None and len(chi2):
        best = float(periods[int(np.argmin(chi2))])

    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.plot(periods, chi2, "-", lw=1.0, color=palette[0], zorder=3)

    # Local minima, deepest first.
    if len(chi2) > 2:
        interior = np.flatnonzero(
            (chi2[1:-1] < chi2[:-2]) & (chi2[1:-1] < chi2[2:])
        ) + 1
        deepest = interior[np.argsort(chi2[interior])][:n_mark]
        ax.plot(
            periods[deepest], chi2[deepest], "o", ms=6.5, mfc="none",
            mec=palette[1], mew=1.5, zorder=4, label=f"{len(deepest)} deepest minima",
        )
    if reference is not None:
        ax.axvline(reference, color=palette[2], lw=1.6, ls=(0, (5, 2)), zorder=2,
                   label=f"Published {reference:.6f} h")
    if best is not None:
        ax.axvline(best, color=palette[1], lw=1.6, ls=(0, (1, 1.6)), zorder=2,
                   label=f"Best {best:.6f} h")
    ax.set_xlabel("Sidereal period $P$ (hours)")
    ax.set_ylabel(r"$\chi^2$")
    ax.set_yscale("log")
    ax.set_xlim(periods.min(), periods.max())
    # Below the axes: a period scan fills its whole panel, so an in-axes legend
    # sits on the data and one above it collides with the title.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=3,
              borderaxespad=0.0)
    ax.set_title("Period scan (Paper II, step 1)", color=tok["text_primary"])
    fig.tight_layout()
    return fig


def plot_pole_samples(lam: np.ndarray, beta: np.ndarray, truth: tuple[float, float] | None = None):
    """Posterior pole directions on an equal-area (Hammer) projection.

    Parameters
    ----------
    lam, beta:
        Posterior samples of the ecliptic longitude and latitude, in degrees.
    truth:
        Optional ``(lambda, beta)`` reference pole to mark.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    tok = _tokens()
    palette = series_colors()
    x = np.radians(np.asarray(lam) % 360.0) - np.pi
    y = np.radians(np.asarray(beta))
    fig, ax = plt.subplots(figsize=(8.2, 4.3), subplot_kw={"projection": "hammer"})
    ax.scatter(x, y, s=5, alpha=0.18, color=palette[0], edgecolors="none", zorder=3)
    if truth is not None:
        ax.scatter(
            [np.radians(truth[0] % 360.0) - np.pi], [np.radians(truth[1])],
            s=200, marker="*", color=palette[1], edgecolors=tok["surface"],
            linewidths=0.8, label="Reference", zorder=5,
        )
        ax.legend(loc="lower right")
    ax.grid(True, lw=0.6, color=tok["grid"])
    ax.tick_params(labelsize=8, colors=tok["text_muted"])
    ax.set_title("Spin-axis posterior (ecliptic)", color=tok["text_primary"], pad=16)
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
    tok = _tokens()
    palette = series_colors()
    cols = [result.labels.index(n) for n in parameters]
    return corner.corner(
        result.samples[:, cols], labels=parameters, truths=truths,
        show_titles=True, title_fmt=".4f",
        title_kwargs={"fontsize": 9, "color": tok["text_primary"]},
        label_kwargs={"fontsize": 10, "color": tok["text_secondary"]},
        color=palette[0], truth_color=palette[1],
        hist_kwargs={"color": palette[0], "lw": 1.6},
        plot_datapoints=False, fill_contours=True, smooth=0.9,
        levels=(0.393, 0.865, 0.989),   # 1, 2, 3 sigma in 2-D
        contour_kwargs={"linewidths": 1.0},
    )


def plot_facet_values(geometry, areas: np.ndarray, title: str = "Curvature function"):
    """The solved ``g_j`` as a map over the Gaussian image sphere.

    This is the quantity the inversion actually determines; the shape follows
    from it only after Minkowski minimisation.

    Parameters
    ----------
    geometry:
        The normal directions the values belong to.
    areas:
        ``(M,)`` solved facet values ``g_j``; divided by the sphere areas to
        recover ``G`` itself.
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from .geometry import unit_to_spherical

    tok = _tokens()
    theta, phi = unit_to_spherical(geometry.normals)
    value = np.asarray(areas) / geometry.sphere_areas
    fig, ax = plt.subplots(figsize=(8.6, 4.2), subplot_kw={"projection": "hammer"})
    sc = ax.scatter(
        np.radians(np.degrees(phi) % 360.0) - np.pi, np.pi / 2 - theta,
        c=value, s=26, cmap=sequential_cmap(), edgecolors="none", zorder=3,
    )
    bar = fig.colorbar(sc, ax=ax, pad=0.03, shrink=0.82)
    bar.set_label(r"Curvature function $G$", color=tok["text_secondary"])
    bar.ax.tick_params(labelsize=8, colors=tok["text_muted"])
    bar.outline.set_visible(False)
    ax.grid(True, lw=0.6, color=tok["grid"])
    ax.tick_params(labelsize=8, colors=tok["text_muted"])
    ax.set_title(title, color=tok["text_primary"], pad=16)
    fig.tight_layout()
    return fig


def plot_pole_scan(scan, references=None, labels=None):
    """Map the pole-scan minima on a Hammer projection, coloured by chi-squared.

    Step 2 of the Paper II recipe starts the fit from a grid of pole directions
    because the ``chi^2`` surface is multimodal.  This is the figure that shows
    whether a "pole disagreement" is an error or a genuine degeneracy: a
    lightcurve data set typically supports two solutions about
    :math:`180^\\circ` apart in longitude, and if their ``chi^2`` differ by less
    than the noise, neither is preferred.

    Parameters
    ----------
    scan:
        Sequence of ``(chi2, lambda_deg, beta_deg)`` from the fitted minima.
    references:
        Optional ``[(lambda, beta), ...]`` reference poles to mark.
    labels:
        Names for those references.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    arr = np.asarray([(c, l, b) for c, l, b in scan], dtype=float)
    chi2, lam, beta = arr[:, 0], arr[:, 1], arr[:, 2]
    order = np.argsort(-chi2)  # draw the good ones last, on top
    chi2, lam, beta = chi2[order], lam[order], beta[order]

    fig, ax = plt.subplots(figsize=(9.0, 4.6), subplot_kw={"projection": "hammer"})
    x = np.radians(lam % 360.0) - np.pi
    y = np.radians(beta)
    best = chi2.min()
    sc = ax.scatter(
        # Ramp reversed so the *best* minima are the darkest marks: the light
        # end of a sequential ramp recedes into the surface, and here the small
        # values are the ones the reader is looking for.
        x, y, c=chi2 / best, s=95, cmap=sequential_cmap(reverse=True), zorder=3,
        edgecolors=_tokens()["axis"], linewidths=0.6,
        norm=plt.matplotlib.colors.LogNorm(),
    )
    bar = fig.colorbar(sc, ax=ax, pad=0.03, shrink=0.82)
    bar.set_label(r"$\chi^2 / \chi^2_{\mathrm{best}}$",
                  color=_tokens()["text_secondary"])
    bar.ax.tick_params(labelsize=8, colors=_tokens()["text_muted"])
    bar.outline.set_visible(False)

    markers = ["*", "P", "X", "D"]
    for i, ref in enumerate(references or []):
        name = (labels or [None] * len(references))[i] or f"reference {i + 1}"
        ax.scatter(
            [np.radians(ref[0] % 360.0) - np.pi], [np.radians(ref[1])],
            s=230, marker=markers[i % len(markers)], color=series_colors()[1],
            edgecolors=_tokens()["surface"], linewidths=0.9, zorder=6, label=name,
        )
    if references:
        ax.legend(loc="lower right")
    ax.grid(True, lw=0.6, color=_tokens()["grid"])
    ax.tick_params(labelsize=8, colors=_tokens()["text_muted"])
    ax.set_title(
        "Pole-scan minima (Paper II, step 2)", color=_tokens()["text_primary"], pad=16
    )
    fig.tight_layout()
    return fig


def plot_phase_function(laws, labels, alpha_max_deg=25.0):
    """Compare solar phase functions ``f(alpha)`` over the observed range.

    Fitting ``a``, ``d`` and ``k`` rather than fixing them is what Paper II's
    step 3 adds; this shows how far the fitted function moves from the
    defaults, and therefore how much of the shape solution was being absorbed
    by the scattering model.

    Parameters
    ----------
    laws:
        Scattering laws to compare; those without a phase function are skipped.
    labels:
        One name per law, used for both the legend and the direct labels.
    alpha_max_deg:
        Upper end of the plotted phase-angle range.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    alpha = np.linspace(0.0, np.radians(alpha_max_deg), 300)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for law, name in zip(laws, labels, strict=True):
        pf = getattr(law, "phase_function", None)
        if pf is None:
            continue
        ax.plot(np.degrees(alpha), pf(alpha), lw=1.8, label=name)
    ax.set_xlabel(r"Solar phase angle $\alpha$ (degrees)")
    ax.set_ylabel(r"$f(\alpha)$")
    ax.set_title("Empirical solar phase function", color=_tokens()["text_primary"])
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_residuals(data, models, bins: int = 40):
    """Residual distribution and per-curve scatter against phase angle.

    The left panel checks that the residuals are noise-like; the right one
    checks Section 3.5's concern about observing geometry, by showing whether
    the fit degrades systematically at the phase angles that carry the shape
    information.

    Parameters
    ----------
    data:
        The observations.
    models:
        Model brightnesses per curve, normalised the same way as the data.
    bins:
        Histogram bin count for the pooled residuals.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    per_curve, alphas, counts = [], [], []
    pooled = []
    for i, curve in enumerate(data):
        res = np.asarray(models[i]) - curve.normalised
        pooled.append(res)
        per_curve.append(float(np.sqrt(np.mean(res**2))))
        alphas.append(np.degrees(curve.mean_phase_angle))
        counts.append(len(curve))
    pooled = np.concatenate(pooled)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    axes[0].hist(pooled, bins=bins, color=series_colors()[0],
                 edgecolor=_tokens()["surface"], linewidth=0.6, zorder=3)
    axes[0].axvline(0.0, color=_tokens()["text_muted"], lw=1.0, zorder=4)
    axes[0].set_xlabel("Residual (relative intensity)")
    axes[0].set_ylabel("Number of points")
    axes[0].set_title(rf"Pooled residuals, RMS $= {np.sqrt(np.mean(pooled**2)):.4f}$",
                      color=_tokens()["text_primary"])

    sizes = 18.0 + 90.0 * np.asarray(counts) / max(counts)
    axes[1].axhline(float(np.median(per_curve)), color=_tokens()["text_muted"],
                    lw=1.0, ls=(0, (5, 2)), zorder=2,
                    label=f"Median {np.median(per_curve):.4f}")
    axes[1].scatter(alphas, per_curve, s=sizes, alpha=0.85, color=series_colors()[1],
                    edgecolors=_tokens()["surface"], linewidths=0.7, zorder=3)
    axes[1].set_xlabel(r"Mean solar phase angle $\alpha$ (degrees)")
    axes[1].set_ylabel("Per-curve RMS")
    axes[1].set_title("Fit quality vs. observing geometry", color=_tokens()["text_primary"])
    axes[1].legend(loc="best")
    fig.tight_layout()
    return fig


def _weight_axis(ax, weights: np.ndarray) -> None:
    """Log weight axis ticked at the weights actually run, not at the decades."""
    from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(weights)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_xlabel(r"Regularisation weight $\gamma$")


def plot_regularisation_scan(
    weights,
    chi2,
    penalty,
    nonconvexity,
    baseline: float | None = None,
    baseline_label: str = "Convex body, ray traced",
    mode: str = "light",
):
    r"""What the convexity regulariser costs, and what it buys (Section 4).

    Section 4 suppresses spurious concavities by penalising "the area 'sunk
    below' the convex hull of the current result".  Turning that weight up
    walks the solution back towards convexity, and the honest test of whether
    the concavities were earning their keep is to watch :math:`\chi^2` while it
    happens - hence two panels against the same weight axis:

    * the fit quality, against Paper II's bar ("at least as good a
      :math:`\chi^2` as that of the convex model"), drawn as ``baseline`` with
      the region that clears it shaded;
    * how much concavity survives, as a fraction of the unregularised
      solution's.  The sunk area and the volume deficit are quantities of
      different scale, so they are indexed to a common base rather than given
      an axis each; the legend carries their absolute values.

    ``weights`` may include the unregularised run.  Zero cannot sit on a
    logarithmic axis and is the natural reference for the rest anyway, so it is
    lifted out and drawn as a reference line.

    Parameters
    ----------
    weights:
        Regularisation weights, as passed to
        :class:`~lcinv.nonconvex.NonconvexInversion`.
    chi2, penalty:
        :attr:`~lcinv.nonconvex.NonconvexResult.chi2` and
        :attr:`~lcinv.nonconvex.NonconvexResult.convexity_penalty` per weight.
    nonconvexity:
        Volume deficit ``1 - V / V_hull`` per weight, as a fraction.
    baseline:
        The convex model's :math:`\chi^2` *through the same forward model* -
        the ray-traced convex body, not the ``L = Ag`` value.
    baseline_label:
        Legend text for that line.
    mode:
        ``"light"`` or ``"dark"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    tok = _tokens(mode)
    palette = series_colors(mode)
    w, chi2, penalty, nonconvexity = (
        np.asarray(a, dtype=float)
        for a in (weights, chi2, penalty, nonconvexity)
    )
    order = np.argsort(w)
    w, chi2, penalty, nonconvexity = (a[order] for a in (w, chi2, penalty, nonconvexity))

    keep = w > 0.0
    if not keep.any():
        raise ValueError("need at least one positive regularisation weight to plot")
    # The reference the other runs are read against: the unregularised solution
    # when it is in the scan, otherwise the weakest regularisation in it.
    free = int(np.argmin(w))
    unregularised = w[free] <= 0.0
    ref_label = r"$\gamma = 0$" if unregularised else rf"$\gamma = {w[free]:g}$"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))

    # --- what it costs -----------------------------------------------------
    ax = axes[0]
    values = [chi2.min(), chi2.max()] + ([baseline] if baseline is not None else [])
    lo, hi = float(min(values)), float(max(values))
    pad = 0.08 * (hi - lo) if hi > lo else 0.1 * abs(hi) + 1e-3
    ax.set_ylim(lo - pad, hi + pad)
    if baseline is not None:
        # Paper II's reliability condition is an inequality, so show it as a
        # region rather than leaving the reader to work out which side is which.
        ax.axhspan(lo - pad, baseline, color=palette[2], alpha=0.14, lw=0, zorder=1)
        ax.axhline(baseline, color=palette[2], lw=1.4, ls=(0, (5, 2)), zorder=3,
                   label=f"{baseline_label} ({baseline:.4f})")
        # Low in the band, clear of the unregularised line that runs through it.
        ax.text(0.985, (lo - pad) + 0.2 * (baseline - lo + pad),
                "at least as good as convex", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5, color=tok["text_secondary"],
                zorder=5, bbox={"facecolor": tok["surface"], "edgecolor": "none",
                                "pad": 1.5, "alpha": 0.85})
    if unregularised:
        ax.axhline(chi2[free], color=tok["text_muted"], lw=1.0, ls=(0, (1, 1.8)),
                   zorder=2, label=rf"Unregularised ({chi2[free]:.4f})")
    ax.plot(w[keep], chi2[keep], "-o", color=palette[0], ms=5.5,
            mec=tok["surface"], mew=0.8, zorder=4, label="Nonconvex fit")
    _weight_axis(ax, w[keep])
    ax.set_ylabel(r"$\chi^2$")
    ax.set_title("What it costs", color=tok["text_primary"])
    ax.legend(loc="upper left")

    # --- what it buys ------------------------------------------------------
    ax = axes[1]
    for series, colour, marker, label in (
        (penalty, palette[0], "o",
         f"Area sunk below the hull ({penalty[free]:.3f} at {ref_label})"),
        (nonconvexity, palette[1], "s",
         f"Volume deficit ({nonconvexity[free]:.1%} at {ref_label})"),
    ):
        scale = series[free] if series[free] > 0 else 1.0
        ax.plot(w[keep], series[keep] / scale, "-", marker=marker, color=colour,
                ms=5.5, mec=tok["surface"], mew=0.8, zorder=4, label=label)
    _weight_axis(ax, w[keep])
    ax.set_yscale("log")
    ax.set_ylabel(f"Fraction of the {ref_label} value")
    ax.set_title("What it buys", color=tok["text_primary"])
    ax.legend(loc="lower left")

    fig.suptitle("Convexity regularisation (Paper I, Section 4)",
                 color=tok["text_primary"])
    fig.tight_layout()
    return fig


def plot_shape_row(
    bodies: list[Polyhedron],
    labels: list[str] | None = None,
    view: np.ndarray | None = None,
    title: str = "",
    colour: str | None = None,
    sun: np.ndarray | None = None,
    mode: str = "light",
):
    """One orthographic view of each of several bodies, side by side.

    For sequences where the shapes are the comparison - a regularisation scan,
    a set of restarts - rather than one body seen from three directions, which
    is :func:`plot_shape_views`.  All panels share one scale, so a body that
    shrinks or flattens across the row reads as such.

    Parameters
    ----------
    bodies:
        The polyhedra, in the order they should appear.  Scale them the same
        way first (``to_unit_volume``) unless size *is* the comparison.
    labels:
        Panel titles.
    view:
        View direction, ``+y`` (equatorial) by default.
    title:
        Figure title.
    colour:
        Base facet colour.
    sun:
        Illumination direction; the view direction when omitted.
    mode:
        ``"light"`` or ``"dark"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if not bodies:
        raise ValueError("nothing to draw")
    tok = _tokens(mode)
    colour = colour or tok["body"]
    view = _VIEWS[1] if view is None else np.asarray(view, dtype=float)
    labels = list(labels) if labels is not None else [""] * len(bodies)

    fig, axes = plt.subplots(1, len(bodies), figsize=(2.3 * len(bodies) + 0.6, 2.7))
    span = max(float(np.abs(b.vertices).max()) for b in bodies) * 1.08
    base = np.array(plt.matplotlib.colors.to_rgb(colour))
    for ax, body, label in zip(np.atleast_1d(axes), bodies, labels, strict=True):
        polys, lit = _shade(body, view, sun)
        shades = base[None, :] * (0.25 + 0.75 * lit)[:, None]
        ax.add_collection(
            PolyCollection(
                polys, facecolors=shades, edgecolors=shades,
                linewidths=0.5, antialiaseds=True,
            )
        )
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9.5, color=tok["text_secondary"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    if title:
        fig.suptitle(title, color=tok["text_primary"])
    fig.tight_layout()
    return fig
