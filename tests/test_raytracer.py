"""Section 2 - the nonconvex direct problem."""

from __future__ import annotations

import numpy as np
import pytest

from lcinv import LommelSeeligerLambert, RayTracer, ellipsoid, paper_shape, sphere
from lcinv.raytracer import hexagonal_facet_samples

LAW = LommelSeeligerLambert(1.0)
RNG = np.random.default_rng(0)
DIRECTIONS = RNG.normal(size=(16, 3))
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)


def brute_force_fraction(body, earth, sun, eps=1e-9):
    """Reference tracer: every facet tested against every other, no culling."""
    normals, centroids = body.normals, body.facet_centroids
    potential = (normals @ earth > 0) & (normals @ sun > 0)
    tri = body.vertices[body.facets]
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = potential.astype(float)
    for j in np.flatnonzero(potential):
        origin = centroids[j] + 8 * eps * normals[j]
        for d in (earth, sun):
            pv = np.cross(d, e2)
            det = np.einsum("ij,ij->i", e1, pv)
            ok = np.abs(det) > 1e-14
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = origin - v0
            u = np.einsum("ij,ij->i", tv, pv) * inv
            qv = np.cross(tv, e1)
            v = (qv @ d) * inv
            t = np.einsum("ij,ij->i", qv, e2) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
            hit[j] = False
            if hit.any():
                out[j] = 0.0
                break
    return out


class TestConvexBodies:
    def test_every_facet_is_a_hull_facet(self):
        """"The facets for which no vertices appear above the local horizon
        belong to the convex hull"."""
        tracer = RayTracer(ellipsoid(2.0, 1.3, 1.0, 8))
        assert tracer.hull_facet_mask.all()
        assert tracer.n_blocker_pairs == 0

    def test_brightness_equals_the_plain_sum_over_facets(self):
        body = ellipsoid(2.0, 1.3, 1.0, 8)
        tracer = RayTracer(body)
        for i in range(0, 16, 2):
            e, s = DIRECTIONS[i], DIRECTIONS[i + 1]
            direct = float((LAW(body.normals @ e, body.normals @ s) * body.areas).sum())
            assert tracer.brightness(e, s, LAW) == pytest.approx(direct)


@pytest.mark.parametrize("name", ["peanut", "castalia", "binary"])
def test_local_blockers_reproduce_brute_force(name):
    """The Section 2 shortcut must not change the answer."""
    body = paper_shape(name, n_rows=6)
    tracer = RayTracer(body)
    for i in range(0, 12, 2):
        e, s = DIRECTIONS[i], DIRECTIONS[i + 1]
        assert tracer.visible_illuminated_fraction(e, s) == pytest.approx(
            brute_force_fraction(body, e, s)
        )


def test_a_concavity_actually_shadows():
    body = paper_shape("peanut", n_rows=6)
    tracer = RayTracer(body)
    shadowed = 0
    for i in range(0, 16, 2):
        e, s = DIRECTIONS[i], DIRECTIONS[i + 1]
        frac = tracer.visible_illuminated_fraction(e, s)
        potential = (body.normals @ e > 0) & (body.normals @ s > 0)
        shadowed += int((potential & (frac == 0)).sum())
    assert shadowed > 0


def test_binary_components_eclipse_each_other():
    """"Binary objects can be handled with the same code"."""
    body = paper_shape("binary", n_rows=6)
    tracer = RayTracer(body)
    along = np.array([1.0, 0.0, 0.0])
    almost = np.array([0.99, 0.14, 0.0])
    almost /= np.linalg.norm(almost)
    frac = tracer.visible_illuminated_fraction(along, almost)
    potential = (body.normals @ along > 0) & (body.normals @ almost > 0)
    assert (potential & (frac == 0)).sum() > 0
    # Viewed down the pole the components cannot hide each other.
    pole = np.array([0.0, 0.0, 1.0])
    assert not ((body.normals @ pole > 0) & (tracer.visible_illuminated_fraction(pole, pole) == 0)).any()


def test_shadowing_only_ever_removes_light():
    body = paper_shape("castalia", n_rows=6)
    tracer = RayTracer(body)
    for i in range(0, 8, 2):
        e, s = DIRECTIONS[i], DIRECTIONS[i + 1]
        traced = tracer.brightness(e, s, LAW)
        unshadowed = float((LAW(body.normals @ e, body.normals @ s) * body.areas).sum())
        assert traced <= unshadowed + 1e-12


def test_hull_lightcurve_is_brighter_than_the_body():
    """Section 3.5: "the projected area of the convex hull is larger than that
    of the original surface"."""
    body = paper_shape("peanut", n_rows=7)
    hull = body.convex_hull()
    angle = np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
    earth = np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])
    sun = np.column_stack([np.cos(angle + 0.5), np.sin(angle + 0.5), np.zeros_like(angle)])
    body_curve = RayTracer(body).lightcurve(earth, sun, LAW)
    hull_curve = RayTracer(hull).lightcurve(earth, sun, LAW)
    assert body_curve.mean() < hull_curve.mean()


class TestTestPoints:
    def test_barycentric_coordinates_are_valid(self):
        for n in (1, 3, 7, 20):
            bary = hexagonal_facet_samples(n)
            assert len(bary) >= n
            assert bary.sum(axis=1) == pytest.approx(np.ones(len(bary)))
            assert (bary > 0).all()

    def test_single_point_is_the_centroid(self):
        assert hexagonal_facet_samples(1) == pytest.approx(np.full((1, 3), 1 / 3))

    def test_subpoints_give_partial_shadowing(self):
        body = paper_shape("castalia", n_rows=5)
        coarse = RayTracer(body, n_subpoints=1)
        fine = RayTracer(body, n_subpoints=10)
        assert fine.n_subpoints >= 10
        seen_partial = False
        for i in range(0, 16, 2):
            frac = fine.visible_illuminated_fraction(DIRECTIONS[i], DIRECTIONS[i + 1])
            seen_partial |= bool(((frac > 0) & (frac < 1)).any())
        assert seen_partial
        # Coarse can only ever be 0 or 1.
        frac = coarse.visible_illuminated_fraction(DIRECTIONS[0], DIRECTIONS[1])
        assert set(np.unique(frac)).issubset({0.0, 1.0})


def test_lightcurve_length_must_match():
    tracer = RayTracer(sphere(1.0, 4))
    with pytest.raises(ValueError):
        tracer.lightcurve(DIRECTIONS[:3], DIRECTIONS[:2], LAW)


def test_isotropic_sphere_has_a_flat_lightcurve():
    body = sphere(1.0, 12)
    tracer = RayTracer(body)
    angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    earth = np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])
    curve = tracer.lightcurve(earth, earth, LAW)
    assert curve.std() / curve.mean() < 2e-3
