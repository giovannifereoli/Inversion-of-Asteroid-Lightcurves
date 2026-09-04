"""Asteroid lightcurve inversion.

A complete implementation of

    M. Kaasalainen and J. Torppa (2001), "Optimization Methods for Asteroid
    Lightcurve Inversion. I. Shape Determination", Icarus 153, 24-36,

together with the recipe of its companion paper (Kaasalainen, Torppa &
Muinonen 2001, Icarus 153, 37-51) that turns it into a complete inverse
problem, a Bayesian sampler for the same forward model, and a client for the
DAMIT database so the whole thing can be run on real photometry.

Where things live
-----------------

=========================================  ===========================================
Paper                                       Module
=========================================  ===========================================
Appendix A, octant triangulation            :mod:`lcinv.triangulation`
Appendix B, gift-wrapping convex hull       :mod:`lcinv.convexhull`
Appendix C, Minkowski minimisation          :mod:`lcinv.minkowski`
Eq. (1), scattering laws                    :mod:`lcinv.scattering`
Section 2, nonconvex direct problem         :mod:`lcinv.raytracer`
Section 3.1, facet areas + conjugate grad.  :class:`lcinv.convex.FacetInversion`
Section 3.2, harmonics + Levenberg-Marq.    :class:`lcinv.convex.HarmonicInversion`
Section 3.3, albedo separation              :mod:`lcinv.albedo`
Sections 3.1/3.4, the chi-squared forms     :class:`lcinv.convex.Objective`
Section 3.5, the four test bodies           :mod:`lcinv.shapes`
Section 4, nonconvex inversion              :mod:`lcinv.nonconvex`
Paper II, the eight-step recipe             :mod:`lcinv.pipeline`
Bayesian version                            :mod:`lcinv.bayes`
=========================================  ===========================================

A minimal run::

    from lcinv import DamitClient, FacetGeometry, InversionPipeline

    client = DamitClient()
    model, data = client.bundle(4966)          # (269) Justitia
    pipeline = InversionPipeline(data)
    result = pipeline.run(model.spin)
    print(result.report())
"""

from __future__ import annotations

__version__ = "0.1.0"

from .albedo import AlbedoResult, AlbedoSeparation
from .convex import (
    ConvexModel,
    FacetGeometry,
    FacetInversion,
    HarmonicInversion,
    InversionResult,
    Objective,
    nonconvexity_residual,
)
from .convexhull import convex_hull, gift_wrap_hull
from .damit import DamitClient, DamitModel, compare_poles
from .geometry import SpinState, phase_angle
from .lightcurve import Lightcurve, LightcurveSet, optimal_scale
from .mesh import Polyhedron
from .minkowski import MinkowskiResult, minkowski_solve
from .nonconvex import (
    CylindricalShapeSeries,
    NonconvexInversion,
    NonconvexResult,
    RadialShapeSeries,
    convexity_penalty,
)
from .style import PALETTE, style_context, use_style
from .pipeline import (
    InversionPipeline,
    PipelineResult,
    period_sampling_interval,
    period_scan,
    pole_grid,
)
from .raytracer import RayTracer
from .scattering import (
    Hapke,
    Lambert,
    LommelSeeliger,
    LommelSeeligerLambert,
    PhaseFunction,
    make_scattering_law,
)
from .shapes import (
    PAPER_SHAPE_NAMES,
    binary,
    castalia_like,
    ellipsoid,
    gaussian_random_sphere,
    irregular_shape,
    paper_shape,
    peanut,
    sphere,
)
from .triangulation import octant_triangulation

__all__ = [
    "__version__",
    # data
    "Lightcurve",
    "LightcurveSet",
    "optimal_scale",
    "DamitClient",
    "DamitModel",
    "compare_poles",
    # geometry
    "SpinState",
    "phase_angle",
    "Polyhedron",
    "octant_triangulation",
    "convex_hull",
    "gift_wrap_hull",
    # physics
    "Lambert",
    "LommelSeeliger",
    "LommelSeeligerLambert",
    "Hapke",
    "PhaseFunction",
    "make_scattering_law",
    "RayTracer",
    # shapes
    "sphere",
    "ellipsoid",
    "peanut",
    "binary",
    "castalia_like",
    "irregular_shape",
    "gaussian_random_sphere",
    "paper_shape",
    "PAPER_SHAPE_NAMES",
    # inversion
    "Objective",
    "FacetGeometry",
    "ConvexModel",
    "InversionResult",
    "HarmonicInversion",
    "FacetInversion",
    "nonconvexity_residual",
    "AlbedoSeparation",
    "AlbedoResult",
    "NonconvexInversion",
    "NonconvexResult",
    "RadialShapeSeries",
    "CylindricalShapeSeries",
    "convexity_penalty",
    "minkowski_solve",
    "MinkowskiResult",
    "InversionPipeline",
    "PipelineResult",
    "PALETTE",
    "use_style",
    "style_context",
    "period_sampling_interval",
    "period_scan",
    "pole_grid",
]


def __getattr__(name: str):
    """Lazily expose the optional Bayesian and plotting layers.

    ``importlib`` is used rather than ``from . import x``: the latter consults
    the parent package's ``__getattr__`` while resolving the submodule, which
    would re-enter this function.
    """
    import importlib

    if name in ("BayesianInversion", "BayesResult"):
        return getattr(importlib.import_module(".bayes", __name__), name)
    if name in ("plotting", "bayes", "cli"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
