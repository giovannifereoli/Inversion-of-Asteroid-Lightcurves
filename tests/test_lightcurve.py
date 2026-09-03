"""Data containers, the paper's normalisations, and DAMIT file formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lcinv import Lightcurve, LightcurveSet, SpinState, optimal_scale
from lcinv.lightcurve import SPEED_OF_LIGHT_AU_PER_DAY

REFERENCE = Path(__file__).parent / "data" / "test_lcs_rel.txt"


@pytest.fixture(scope="module")
def data():
    return LightcurveSet.from_damit_txt(REFERENCE)


def make_curve(n=12, seed=0):
    rng = np.random.default_rng(seed)
    jd = 2450000.0 + np.linspace(0, 0.2, n)
    sun = np.tile([1.2, -0.4, 0.1], (n, 1))
    earth = np.tile([0.6, -0.9, 0.05], (n, 1))
    return Lightcurve(jd, 1.0 + 0.2 * rng.random(n), sun, earth, name="c")


class TestLightcurve:
    def test_normalisation_has_unit_mean(self):
        curve = make_curve()
        assert curve.normalised.mean() == pytest.approx(1.0)
        assert curve.mean_brightness == pytest.approx(curve.brightness.mean())

    def test_phase_angle_is_the_sun_earth_angle(self):
        curve = make_curve()
        s, e = curve.unit_vectors()
        expected = np.arccos(np.clip(np.einsum("ij,ij->i", s, e), -1, 1))
        assert curve.phase_angles == pytest.approx(expected)

    def test_amplitude_in_magnitudes(self):
        curve = make_curve()
        assert curve.amplitude_mag == pytest.approx(
            2.5 * np.log10(curve.brightness.max() / curve.brightness.min())
        )

    def test_body_directions_are_unit_vectors(self):
        curve = make_curve()
        spin = SpinState(30.0, 20.0, 5.0, 2450000.0, 0.0)
        s, e = curve.body_directions(spin)
        assert np.linalg.norm(s, axis=1) == pytest.approx(np.ones(len(curve)))
        assert np.linalg.norm(e, axis=1) == pytest.approx(np.ones(len(curve)))

    def test_light_time_correction_shifts_epochs_back(self):
        curve = make_curve()
        corrected = curve.light_time_corrected()
        delta = np.linalg.norm(curve.earth, axis=1) / SPEED_OF_LIGHT_AU_PER_DAY
        assert corrected.jd == pytest.approx(curve.jd - delta)

    def test_noise_is_reproducible_and_positive(self):
        curve = make_curve()
        a = curve.with_noise(0.05, seed=1)
        b = curve.with_noise(0.05, seed=1)
        assert a.brightness == pytest.approx(b.brightness)
        assert np.all(a.brightness > 0)
        assert not np.allclose(a.brightness, curve.brightness)

    def test_rejects_inconsistent_input(self):
        with pytest.raises(ValueError):
            Lightcurve([1.0, 2.0], [1.0], np.ones((2, 3)), np.ones((2, 3)))
        with pytest.raises(ValueError):
            Lightcurve([1.0, 2.0], [1.0, 1.0], np.ones((2, 2)), np.ones((2, 3)))
        with pytest.raises(ValueError):
            Lightcurve([1.0], [-1.0], np.ones((1, 3)), np.ones((1, 3)))


class TestLightcurveSet:
    def test_reads_the_damit_plaintext_export(self, data):
        assert len(data) == 37
        assert data.n_points == 1541
        assert data.counts.sum() == data.n_points
        assert data.offsets[-1] == data.n_points

    def test_summary_reports_what_section_3_5_asks_for(self, data):
        summary = data.summary()
        assert summary["n_curves"] == 37
        assert summary["phase_max_deg"] > 20.0
        assert summary["n_above_20deg"] > 0

    def test_round_trips_through_the_plaintext_format(self, data, tmp_path):
        path = tmp_path / "lc.txt"
        data.to_damit_txt(path)
        back = LightcurveSet.from_damit_txt(path)
        assert len(back) == len(data)
        for a, b in zip(data, back):
            assert a.jd == pytest.approx(b.jd, abs=1e-6)
            assert a.brightness == pytest.approx(b.brightness, rel=1e-5)
            assert a.calibrated == b.calibrated

    def test_filtering(self, data):
        assert len(data.filter(min_points=40)) < len(data)
        assert all(len(c) >= 40 for c in data.filter(min_points=40))
        high = data.filter(min_phase_deg=20.0)
        assert all(np.degrees(c.mean_phase_angle) >= 20.0 for c in high)
        assert len(data.filter(calibrated=True)) == 0

    def test_geometry_selection_spreads_out(self, data):
        chosen = data.select_geometries(8)
        assert len(chosen) == 8
        # The chosen curves should span more phase angle than the first eight.
        first = np.degrees([c.mean_phase_angle for c in data[:8]])
        picked = np.degrees([c.mean_phase_angle for c in chosen])
        assert np.ptp(picked) >= 0.8 * np.ptp(first)

    def test_selection_is_a_no_op_when_asking_for_everything(self, data):
        assert len(data.select_geometries(999)) == len(data)

    def test_indexing_and_iteration(self, data):
        assert isinstance(data[0], Lightcurve)
        assert isinstance(data[:3], LightcurveSet)
        assert len(list(data)) == len(data)

    def test_body_directions_concatenate(self, data):
        spin = SpinState(253.3, -16.9, 5.761982, 2438882.0, 0.0)
        sun, earth = data.body_directions(spin)
        assert sun.shape == (data.n_points, 3)
        assert earth.shape == (data.n_points, 3)

    def test_appending(self):
        s = LightcurveSet()
        assert len(s) == 0 and s.n_points == 0
        s.append(make_curve())
        assert len(s) == 1


class TestOptimalScale:
    def test_recovers_a_known_factor(self):
        """Eq. (14)."""
        model = np.array([1.0, 2.0, 3.0, 4.0])
        assert optimal_scale(3.7 * model, model) == pytest.approx(3.7)

    def test_is_the_least_squares_minimiser(self):
        rng = np.random.default_rng(0)
        model = rng.random(50) + 0.5
        observed = 2.0 * model + 0.05 * rng.normal(size=50)
        c = optimal_scale(observed, model)
        base = np.sum((observed - c * model) ** 2)
        for delta in (-0.01, 0.01):
            assert np.sum((observed - (c + delta) * model) ** 2) > base

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            optimal_scale(np.ones(3), np.ones(4))
        with pytest.raises(ValueError):
            optimal_scale(np.ones(3), np.zeros(3))
