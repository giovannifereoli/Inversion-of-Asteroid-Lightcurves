"""Command-line interface: ``lcinv <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .damit import DamitClient

    client = DamitClient(cache_dir=args.cache)
    model, data = client.bundle(args.model, fmt="json")
    print(model)
    print(f"  scattering law : {model.law}")
    print(f"  lightcurves    : {data}")
    for key, value in data.summary().items():
        print(f"    {key:>18} : {value}")
    print(f"  cached under   : {Path(args.cache).resolve()}")
    return 0


def _cmd_invert(args: argparse.Namespace) -> int:
    from .convex import FacetGeometry
    from .damit import DamitClient, compare_poles
    from .pipeline import InversionPipeline
    from .scattering import LommelSeeligerLambert, PhaseFunction

    client = DamitClient(cache_dir=args.cache)
    model, data = client.bundle(args.model, fmt="txt")
    if args.max_curves:
        data = data.select_geometries(args.max_curves)
    print(f"{model.label}: {data}")

    pipeline = InversionPipeline(
        data,
        geometry=FacetGeometry.from_sphere(args.rows),
        # The papers' own law; DAMIT's published parameters are reported below
        # for comparison but are not used to drive the fit.
        law=LommelSeeligerLambert(0.1, PhaseFunction(0.5, 0.1, -0.5)),
        lmax=args.lmax,
    )
    result = pipeline.run(
        model.spin,
        fit_pole=not args.fix_pole,
        fit_period=not args.fix_period,
        n_restarts=args.restarts,
        reconstruct=not args.no_shape,
        verbose=True,
    )
    print("\n" + result.report())
    print(
        f"\nDAMIT reference: lambda={model.spin.lam:.2f} beta={model.spin.beta:.2f} "
        f"P={model.spin.period:.6f} h  (quality flag {model.quality_flag})"
    )
    print(f"pole difference from DAMIT: {compare_poles(result.best.spin, model.spin):.2f} deg")
    if result.shape is not None and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.shape.polyhedron.to_obj(args.out)
        print(f"shape written to {args.out}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .convex import FacetGeometry, HarmonicInversion
    from .lightcurve import Lightcurve, LightcurveSet
    from .geometry import SpinState
    from .raytracer import RayTracer
    from .scattering import LommelSeeligerLambert
    from .shapes import paper_shape

    body = paper_shape(args.shape)
    hull = body.convex_hull()
    print(f"shape {args.shape}: {body}, nonconvexity {1 - body.volume / hull.volume:.1%}")

    law = LommelSeeligerLambert(1.0)
    tracer = RayTracer(body)
    spin = SpinState(60.0, 25.0, 6.0, 2450000.0, 0.0)
    rng = np.random.default_rng(0)
    curves = []
    for k in range(args.curves):
        jd = 2450000.0 + 40.0 * k + np.linspace(0.0, 6.0 / 24.0, 60)
        lam = rng.uniform(0, 2 * np.pi)
        alpha = np.radians(rng.uniform(15.0, 60.0))
        sun = np.tile([np.cos(lam), np.sin(lam), 0.0], (len(jd), 1))
        earth = np.tile(
            [np.cos(lam + alpha), np.sin(lam + alpha), 0.0], (len(jd), 1)
        )
        s_b = spin.to_asteroid_frame(sun, jd)
        e_b = spin.to_asteroid_frame(earth, jd)
        flux = tracer.lightcurve(e_b, s_b, law)
        curves.append(Lightcurve(jd, flux, sun, earth, name=f"syn{k + 1:02d}"))
    data = LightcurveSet(curves).with_noise(args.noise)
    print(f"generated {data}, noise {args.noise:.1%}")

    geometry = FacetGeometry.from_sphere(args.rows)
    result = HarmonicInversion(data, geometry, spin, lmax=args.lmax, law=law).run()
    print(f"inversion: chi2={result.chi2:.6f} rms={result.rms:.6f}")
    print(f"residual nonconvexity {result.nonconvexity:.4f}")
    shape = result.shape(geometry)
    ex, ex_true = shape.polyhedron.extents(), hull.extents()
    print(f"recovered a:b:c = {ex[0] / ex[2]:.3f} : {ex[1] / ex[2]:.3f} : 1.000")
    print(f"convex hull     = {ex_true[0] / ex_true[2]:.3f} : {ex_true[1] / ex_true[2]:.3f} : 1.000")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    """Entry point for the ``lcinv`` command."""
    parser = argparse.ArgumentParser(
        prog="lcinv",
        description="Asteroid lightcurve inversion (Kaasalainen & Torppa 2001).",
    )
    parser.add_argument("--version", action="version", version=f"lcinv {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache", default="data/damit", help="download cache directory")

    p_fetch = sub.add_parser("fetch", parents=[common], help="download a DAMIT model and its lightcurves")
    p_fetch.add_argument("model", type=int, help="DAMIT model id, e.g. 4966")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_inv = sub.add_parser("invert", parents=[common], help="run the full inversion on a DAMIT target")
    p_inv.add_argument("model", type=int, help="DAMIT model id, e.g. 4966")
    p_inv.add_argument("--rows", type=int, default=7, help="octant rows (facets = 8 N^2)")
    p_inv.add_argument("--lmax", type=int, default=6, help="harmonic truncation degree")
    p_inv.add_argument("--max-curves", type=int, default=0, help="use at most this many lightcurves")
    p_inv.add_argument("--restarts", type=int, default=0, help="perturbed restarts for error estimates")
    p_inv.add_argument("--fix-pole", action="store_true", help="hold the pole fixed")
    p_inv.add_argument("--fix-period", action="store_true", help="hold the period fixed")
    p_inv.add_argument("--no-shape", action="store_true", help="skip Minkowski reconstruction")
    p_inv.add_argument("--out", default="", help="write the recovered shape to this .obj")
    p_inv.set_defaults(func=_cmd_invert)

    p_demo = sub.add_parser("demo", help="synthetic end-to-end check on a Section 3.5 body")
    p_demo.add_argument("--shape", default="peanut", help="1-4 or irregular/castalia/peanut/binary")
    p_demo.add_argument("--curves", type=int, default=10, help="synthetic lightcurves to generate")
    p_demo.add_argument("--noise", type=float, default=0.02, help="relative noise level")
    p_demo.add_argument("--rows", type=int, default=7)
    p_demo.add_argument("--lmax", type=int, default=6)
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
