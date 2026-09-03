"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lcinv import (
    FacetGeometry,
    Lightcurve,
    LightcurveSet,
    LommelSeeligerLambert,
    RayTracer,
    SpinState,
    paper_shape,
)

REFERENCE_DATA = Path(__file__).parent / "data" / "test_lcs_rel.txt"


@pytest.fixture(scope="session")
def law():
    """The paper's equal-weight Lommel-Seeliger + Lambert law (Section 3.5)."""
    return LommelSeeligerLambert(1.0)


@pytest.fixture(scope="session")
def spin():
    return SpinState(lam=60.0, beta=25.0, period=6.0, t0=2450000.0, phi0=0.0)


def synthetic_set(body, spin, law, n_curves=8, n_points=45, seed=0, noise=0.0):
    """Ray-traced lightcurves of ``body`` over varied observing geometries."""
    tracer = RayTracer(body)
    rng = np.random.default_rng(seed)
    curves = []
    for k in range(n_curves):
        jd = spin.t0 + 40.0 * k + np.linspace(0.0, 6.0 / 24.0, n_points)
        lam = rng.uniform(0.0, 2.0 * np.pi)
        # Section 3.5: "observing geometries must reach large solar phase angles".
        alpha = np.radians(rng.uniform(20.0, 60.0))
        tilt = rng.uniform(-0.3, 0.3)
        sun = np.tile([np.cos(lam), np.sin(lam), tilt], (len(jd), 1))
        earth = np.tile([np.cos(lam + alpha), np.sin(lam + alpha), -tilt], (len(jd), 1))
        sun /= np.linalg.norm(sun, axis=1, keepdims=True)
        earth /= np.linalg.norm(earth, axis=1, keepdims=True)
        flux = tracer.lightcurve(
            spin.to_asteroid_frame(earth, jd), spin.to_asteroid_frame(sun, jd), law
        )
        curves.append(Lightcurve(jd, flux, sun, earth, name=f"syn{k + 1:02d}"))
    data = LightcurveSet(curves)
    return data.with_noise(noise, seed=seed) if noise else data


@pytest.fixture(scope="session")
def ellipsoid_data(spin, law):
    from lcinv import ellipsoid

    return synthetic_set(ellipsoid(1.5, 1.0, 0.85, 7), spin, law)


@pytest.fixture(scope="session")
def geometry():
    return FacetGeometry.from_sphere(6)


@pytest.fixture(scope="session")
def peanut_body():
    return paper_shape("peanut", n_rows=7)
