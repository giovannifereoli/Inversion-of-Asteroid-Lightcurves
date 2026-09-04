"""Rotation state, spherical harmonics and polyhedra."""

from __future__ import annotations

import numpy as np
import pytest

from lcinv import Polyhedron, SpinState, ellipsoid, octant_triangulation, phase_angle, sphere
from lcinv.geometry import rot_y, rot_z, spherical_to_unit, unit_to_spherical
from lcinv.sphharm import design_matrix, n_coefficients, real_sph_harm, sph_harm_indices

RNG = np.random.default_rng(1)
SPIN = SpinState(35.0, -12.0, 7.81323, 2433827.77154, 17.0)


class TestSpinState:
    def test_frames_round_trip(self):
        v = RNG.normal(size=(7, 3))
        jd = SPIN.t0 + RNG.uniform(0.0, 50.0, 7)
        back = SPIN.to_ecliptic_frame(SPIN.to_asteroid_frame(v, jd), jd)
        assert back == pytest.approx(v)

    def test_vectorised_matches_scalar(self):
        v = RNG.normal(size=(5, 3))
        jd = SPIN.t0 + RNG.uniform(0.0, 10.0, 5)
        one_by_one = np.array([SPIN.to_asteroid_frame(v[i], jd[i]) for i in range(5)])
        assert SPIN.to_asteroid_frame(v, jd) == pytest.approx(one_by_one)

    def test_matches_the_damit_transformation(self):
        """r_ecl = Rz(lambda) Ry(90 - beta) Rz(phi) r_ast."""
        jd = SPIN.t0 + 3.5
        expected = (
            rot_z(np.radians(SPIN.lam))
            @ rot_y(np.radians(90.0 - SPIN.beta))
            @ rot_z(float(SPIN.rotation_angle(jd)))
        )
        assert SPIN.matrix_ast_to_ecl(jd) == pytest.approx(expected)

    def test_body_z_axis_is_the_spin_axis(self):
        got = SPIN.to_ecliptic_frame(np.array([0.0, 0.0, 1.0]), SPIN.t0 + 9.0)
        assert got == pytest.approx(SPIN.pole_vector())

    def test_one_period_is_a_full_turn(self):
        delta = SPIN.rotation_angle(SPIN.t0 + SPIN.period / 24.0) - SPIN.rotation_angle(SPIN.t0)
        assert delta == pytest.approx(2.0 * np.pi)

    def test_yorp_adds_a_quadratic_term(self):
        yorp = SpinState(35.0, -12.0, 7.81323, SPIN.t0, 0.0, 1e-8)
        plain = SpinState(35.0, -12.0, 7.81323, SPIN.t0, 0.0, 0.0)
        dt = 100.0
        extra = yorp.rotation_angle(SPIN.t0 + dt) - plain.rotation_angle(SPIN.t0 + dt)
        assert extra == pytest.approx(0.5 * 1e-8 * dt**2)

    def test_rotation_preserves_the_phase_angle(self):
        sun = RNG.normal(size=(6, 3))
        earth = RNG.normal(size=(6, 3))
        jd = SPIN.t0 + RNG.uniform(0, 5, 6)
        s = SPIN.to_asteroid_frame(sun / np.linalg.norm(sun, axis=1, keepdims=True), jd)
        e = SPIN.to_asteroid_frame(earth / np.linalg.norm(earth, axis=1, keepdims=True), jd)
        assert phase_angle(s, e) == pytest.approx(phase_angle(sun, earth))

    def test_parameters_round_trip(self):
        assert SPIN.with_parameters(SPIN.parameters).parameters == pytest.approx(SPIN.parameters)


def test_spherical_conversion_round_trip():
    theta, phi = np.array([0.3, 1.2, 2.9]), np.array([0.7, -2.0, 3.0])
    got = unit_to_spherical(spherical_to_unit(theta, phi))
    assert got[0] == pytest.approx(theta)
    assert np.cos(got[1]) == pytest.approx(np.cos(phi))


class TestSphericalHarmonics:
    def test_coefficient_count(self):
        for lmax in (0, 3, 4, 6, 9):
            assert n_coefficients(lmax) == (lmax + 1) ** 2
            assert len(sph_harm_indices(lmax)) == (lmax + 1) ** 2

    def test_closed_forms(self):
        theta, phi = np.array([0.7]), np.array([1.1])
        assert real_sph_harm(0, 0, theta, phi)[0] == pytest.approx(1 / np.sqrt(4 * np.pi))
        assert real_sph_harm(1, 0, theta, phi)[0] == pytest.approx(
            np.sqrt(3 / (4 * np.pi)) * np.cos(0.7)
        )
        # The Condon-Shortley phase makes Y_1^1 negative here.
        assert real_sph_harm(1, 1, theta, phi)[0] == pytest.approx(
            -np.sqrt(3 / (4 * np.pi)) * np.sin(0.7) * np.cos(1.1)
        )

    def test_orthonormal_on_the_sphere(self):
        body = Polyhedron(*_mesh(28))
        centres = body.facet_centroids
        centres = centres / np.linalg.norm(centres, axis=1, keepdims=True)
        weights = body.areas * (4.0 * np.pi / body.surface_area)
        theta, phi = unit_to_spherical(centres)
        basis = design_matrix(5, theta, phi)
        gram = (basis * weights[:, None]).T @ basis
        assert np.abs(gram - np.eye(gram.shape[0])).max() < 2e-3

    def test_design_matrix_agrees_with_single_evaluations(self):
        theta, phi = np.array([0.4, 2.1]), np.array([0.2, -1.3])
        expected = np.column_stack(
            [real_sph_harm(l, m, theta, phi) for l, m in sph_harm_indices(3)]
        )
        assert design_matrix(3, theta, phi) == pytest.approx(expected)

    def test_high_degree_does_not_overflow(self):
        theta, phi = np.linspace(0.01, 3.13, 40), np.linspace(0, 6.0, 40)
        assert np.isfinite(design_matrix(40, theta, phi)).all()

    def test_rejects_bad_orders(self):
        with pytest.raises(ValueError):
            real_sph_harm(1, 2, np.array([0.1]), np.array([0.1]))


def _mesh(n):
    mesh = octant_triangulation(n)
    return mesh.vertices, mesh.facets


class TestPolyhedron:
    def test_sphere_converges_to_analytic_values(self):
        # An inscribed polyhedron under-estimates both, by O(1/N^2).
        body = sphere(1.0, 20)
        assert body.volume == pytest.approx(4 / 3 * np.pi, rel=5e-3)
        assert body.surface_area == pytest.approx(4 * np.pi, rel=5e-3)
        assert body.equivalent_diameter == pytest.approx(2.0, rel=2e-3)
        assert body.volume < 4 / 3 * np.pi
        finer = sphere(1.0, 40)
        assert abs(finer.volume - 4 / 3 * np.pi) < 0.35 * abs(body.volume - 4 / 3 * np.pi)

    def test_ellipsoid_volume_and_extents(self):
        body = ellipsoid(3.0, 2.0, 1.0, 20)
        assert body.volume == pytest.approx(4 / 3 * np.pi * 6.0, rel=5e-3)
        assert body.extents() == pytest.approx([6.0, 4.0, 2.0], rel=1e-6)

    def test_sphere_moments_are_isotropic(self):
        moments, _ = sphere(1.0, 16).principal_axes()
        assert moments.std() / moments.mean() < 1e-2

    def test_ellipsoid_moments_match_theory(self):
        body = ellipsoid(3.0, 2.0, 1.0, 20)
        mass = body.volume
        expected = np.sort(mass / 5.0 * np.array([4 + 1, 9 + 1, 9 + 4]))
        assert body.principal_axes()[0] == pytest.approx(expected, rel=2e-2)

    def test_projected_area_of_a_sphere(self):
        assert sphere(1.0, 20).projected_area([0.0, 0.0, 1.0]) == pytest.approx(np.pi, rel=5e-3)

    def test_closure_identity(self):
        assert np.linalg.norm(ellipsoid(2.0, 1.0, 0.7, 8).facet_normal_sum) < 1e-12

    def test_centering_and_unit_volume(self):
        body = ellipsoid(2.0, 1.0, 0.7, 8).translated([5.0, -3.0, 2.0]).centered().to_unit_volume()
        assert body.volume == pytest.approx(1.0)
        assert body.centroid == pytest.approx(np.zeros(3), abs=1e-12)

    def test_flip_reverses_the_sign_of_the_volume(self):
        body = ellipsoid(2.0, 1.0, 0.7, 6)
        assert body.flipped().volume == pytest.approx(-body.volume)
        assert body.flipped().oriented_outward().volume == pytest.approx(body.volume)

    def test_convex_hull_of_a_convex_body_is_itself(self):
        body = ellipsoid(2.0, 1.0, 0.7, 7)
        assert body.convex_hull().volume == pytest.approx(body.volume, rel=1e-9)

    def test_rejects_degenerate_facets(self):
        with pytest.raises(ValueError):
            Polyhedron(np.zeros((3, 3)), np.array([[0, 1, 2]]))

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            Polyhedron(np.zeros((3, 2)), np.array([[0, 1, 2]]))
        with pytest.raises(ValueError):
            Polyhedron(np.eye(3), np.array([[0, 1, 2, 0]]))

    def test_shape_txt_round_trip(self, tmp_path):
        body = ellipsoid(2.0, 1.0, 0.7, 5)
        path = tmp_path / "shape.txt"
        body.to_shape_txt(path)
        back = Polyhedron.from_shape_txt(path)
        assert back.vertices == pytest.approx(body.vertices, abs=1e-8)
        assert np.array_equal(back.facets, body.facets)

    def test_obj_round_trip(self, tmp_path):
        body = ellipsoid(2.0, 1.0, 0.7, 5)
        path = tmp_path / "shape.obj"
        body.to_obj(path)
        back = Polyhedron.from_obj(path)
        assert back.vertices == pytest.approx(body.vertices, abs=1e-8)
        assert np.array_equal(back.facets, body.facets)

    def test_obj_reader_triangulates_polygons(self, tmp_path):
        path = tmp_path / "quad.obj"
        path.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
        )
        assert Polyhedron.from_obj(path).n_facets == 2

    def test_albedo_broadcasts(self):
        body = sphere(1.0, 4)
        body.albedo = 0.3
        assert body.albedo.shape == (body.n_facets,)
        assert np.all(body.albedo == 0.3)


class TestPoleNormalisation:
    """A fitted pole may wander past the poles; the reported one should not."""

    @pytest.mark.parametrize(
        "lam,beta",
        [(58.47, -93.75), (73.0, -81.0), (200.0, 95.0), (400.0, -10.0), (10.0, -180.0)],
    )
    def test_preserves_the_direction(self, lam, beta):
        raw = SpinState(lam, beta, 5.0)
        assert np.allclose(raw.normalised().pole_vector(), raw.pole_vector(), atol=1e-12)

    @pytest.mark.parametrize(
        "lam,beta", [(58.47, -93.75), (200.0, 95.0), (400.0, -10.0), (-30.0, 100.0)]
    )
    def test_lands_in_the_canonical_range(self, lam, beta):
        n = SpinState(lam, beta, 5.0).normalised()
        assert 0.0 <= n.lam < 360.0
        assert -90.0 <= n.beta <= 90.0

    def test_leaves_a_canonical_pole_alone(self):
        n = SpinState(73.0, -81.0, 5.0, 2450000.0, 12.0).normalised()
        assert n.lam == pytest.approx(73.0)
        assert n.beta == pytest.approx(-81.0)
        assert n.t0 == 2450000.0 and n.phi0 == 12.0
