"""Eq. (1) and the scattering laws."""

from __future__ import annotations

import numpy as np
import pytest

from lcinv import Hapke, Lambert, LommelSeeliger, LommelSeeligerLambert, PhaseFunction
from lcinv.scattering import make_scattering_law

MU = np.array([0.5, 0.8, -0.1, 0.3, 1.0])
MU0 = np.array([0.7, -0.2, 0.4, 0.9, 1.0])
VISIBLE = (MU > 0) & (MU0 > 0)


@pytest.mark.parametrize(
    "law", [Lambert(), LommelSeeliger(), LommelSeeligerLambert(0.1),
            LommelSeeligerLambert(0.1, PhaseFunction()), Hapke()]
)
def test_vanishes_when_not_visible_or_illuminated(law):
    """Eq. (4): "If either mu or mu0 is less than or equal to 0, A_ij vanishes"."""
    alpha = 0.3 if law.uses_phase_angle else None
    value = law(MU, MU0, alpha)
    assert np.all(value[~VISIBLE] == 0.0)
    assert np.all(value[VISIBLE] > 0.0)


def test_lambert_is_mu_mu0():
    assert Lambert()(MU, MU0)[0] == pytest.approx(MU[0] * MU0[0])


def test_lommel_seeliger_is_lambert_over_the_sum():
    """"Lommel-Seeliger law is S_LS = S_L / (mu + mu0)"."""
    expected = Lambert()(MU, MU0)[VISIBLE] / (MU + MU0)[VISIBLE]
    assert LommelSeeliger()(MU, MU0)[VISIBLE] == pytest.approx(expected)


def test_equal_weight_combination_is_the_sum_of_its_parts():
    """Section 3.5's "combination of Lommel-Seeliger and Lambert laws with
    equal weights"."""
    combined = LommelSeeligerLambert(1.0)(MU, MU0)
    assert combined == pytest.approx(LommelSeeliger()(MU, MU0) + Lambert()(MU, MU0))


def test_lambert_weight_scales_only_the_lambert_term():
    c = 0.37
    got = LommelSeeligerLambert(c)(MU, MU0)
    assert got == pytest.approx(LommelSeeliger()(MU, MU0) + c * Lambert()(MU, MU0))


def test_phase_function_form_and_opposition_value():
    """f(alpha) = a exp(-alpha/d) + k alpha + 1."""
    f = PhaseFunction(0.5, 0.1, -0.5)
    assert f(0.0) == pytest.approx(1.5)
    assert f(0.3) == pytest.approx(0.5 * np.exp(-3.0) - 0.15 + 1.0)


def test_phase_function_multiplies_the_whole_law():
    plain = LommelSeeligerLambert(0.1)
    withf = LommelSeeligerLambert(0.1, PhaseFunction(0.5, 0.1, -0.5))
    alpha = 0.25
    assert withf(MU, MU0, alpha) == pytest.approx(plain(MU, MU0) * PhaseFunction(0.5, 0.1, -0.5)(alpha))


def test_phase_function_rejects_bad_width():
    with pytest.raises(ValueError):
        PhaseFunction(0.5, 0.0, -0.5)(0.1)


def test_law_needing_alpha_says_so():
    with pytest.raises(ValueError):
        LommelSeeligerLambert(0.1, PhaseFunction())(MU, MU0)
    with pytest.raises(ValueError):
        Hapke()(MU, MU0)


@pytest.mark.parametrize(
    "law", [LommelSeeligerLambert(0.1), LommelSeeligerLambert(0.1, PhaseFunction()), Hapke()]
)
def test_parameters_round_trip(law):
    assert law.with_parameters(law.parameters) == law


def test_hapke_brightens_with_single_scattering_albedo():
    low = Hapke(w=0.1)(MU, MU0, 0.2)
    high = Hapke(w=0.5)(MU, MU0, 0.2)
    assert np.all(high[VISIBLE] > low[VISIBLE])


def test_make_scattering_law_maps_damit_codes():
    assert make_scattering_law("LSL", [0.1, None, None, None, None]) == LommelSeeligerLambert(0.1)
    lsl = make_scattering_law("LSL", [0.1, 0.5, 0.1, -0.5, None])
    assert lsl.phase_function == PhaseFunction(0.5, 0.1, -0.5)
    hapke = make_scattering_law("H", [0.083, -0.292, 0.056, 1.172, 20.0])
    assert isinstance(hapke, Hapke) and hapke.w == pytest.approx(0.083)
    with pytest.raises(ValueError):
        make_scattering_law("XX")


class TestHapkeRoughness:
    """Hapke's macroscopic-roughness correction (1984; 1993, chapter 12)."""

    @staticmethod
    def _consistent_geometry(n=2000, seed=0):
        """Random (mu, mu0, alpha) obeying |i - e| <= alpha <= i + e."""
        rng = np.random.default_rng(seed)
        mu = rng.uniform(0.05, 1.0, n)
        mu0 = rng.uniform(0.05, 1.0, n)
        i, e = np.arccos(mu0), np.arccos(mu)
        alpha = rng.uniform(0.0, np.pi, n)
        alpha = np.clip(alpha, np.abs(i - e) + 1e-6, np.minimum(i + e, np.pi) - 1e-6)
        return mu, mu0, alpha

    def test_zero_roughness_reproduces_the_smooth_law(self):
        mu, mu0, alpha = self._consistent_geometry()
        smooth = Hapke(0.3, -0.3, 1.0, 0.05, 0.0)(mu, mu0, alpha)
        rough = Hapke(0.3, -0.3, 1.0, 0.05, 1e-12)(mu, mu0, alpha)
        assert np.allclose(smooth, rough, rtol=1e-9, atol=1e-15)

    def test_roughness_darkens_more_at_large_phase_angle(self):
        """The characteristic signature: little effect near opposition, strong
        darkening at large alpha."""
        near = np.full(64, 0.7), np.full(64, 0.7), np.full(64, np.radians(2.0))
        far = np.full(64, 0.5), np.full(64, 0.5), np.full(64, np.radians(60.0))
        ratios = []
        for theta in (0.0, 10.0, 20.0, 40.0):
            law = Hapke(0.083, -0.292, 0.056, 1.172, theta)
            ratios.append((law(*near).mean(), law(*far).mean()))
        smooth_near, smooth_far = ratios[0]
        for near_v, far_v in ratios[1:]:
            assert near_v <= smooth_near * (1.0 + 1e-9)
            assert far_v <= smooth_far * (1.0 + 1e-9)
        # Monotonically darker with roughness, and much more so at large alpha.
        far_values = [r[1] for r in ratios]
        assert far_values == sorted(far_values, reverse=True)
        assert far_values[-1] / smooth_far < 0.75
        assert ratios[-1][0] / smooth_near > 0.95

    @pytest.mark.parametrize("theta", [0.0, 5.0, 20.0, 45.0, 60.0])
    def test_stays_finite_and_non_negative(self, theta):
        mu, mu0, alpha = self._consistent_geometry(seed=3)
        v = Hapke(0.3, -0.3, 1.0, 0.05, theta)(mu, mu0, alpha)
        assert np.all(np.isfinite(v))
        assert np.all(v >= 0.0)

    def test_masks_unlit_and_unseen_facets(self):
        law = Hapke(0.3, -0.3, 1.0, 0.05, 20.0)
        v = law(np.array([-0.1, 0.5, 0.4]), np.array([0.5, -0.1, 0.6]), np.full(3, 0.3))
        assert v[0] == 0.0 and v[1] == 0.0 and v[2] > 0.0

    def test_survives_the_limb(self):
        law = Hapke(0.3, -0.3, 1.0, 0.05, 20.0)
        v = law(np.array([1e-10, 1e-6, 1e-3]), np.full(3, 0.6), np.full(3, 0.5))
        assert np.all(np.isfinite(v)) and np.all(v >= 0.0)


class TestIdentifiabilityAndBounds:
    """A parameter that acts as a pure scale cannot be fitted under Eq. (13)."""

    def test_hapke_holds_the_single_scattering_albedo_fixed(self):
        mask = Hapke().free_parameter_mask
        assert mask[0] is np.False_ or mask[0] == False  # noqa: E712
        assert mask[1:].all()

    def test_lommel_seeliger_lambert_frees_everything(self):
        assert LommelSeeligerLambert(0.1, PhaseFunction()).free_parameter_mask.all()

    @pytest.mark.parametrize(
        "law",
        [Hapke(), LommelSeeligerLambert(0.1, PhaseFunction()), LommelSeeligerLambert(0.1)],
    )
    def test_defaults_lie_inside_the_declared_bounds(self, law):
        lo, hi = law.parameter_bounds
        assert np.all(law.parameters >= lo) and np.all(law.parameters <= hi)

    def test_hapke_bounds_exclude_unphysical_values(self):
        lo, hi = Hapke().parameter_bounds
        assert lo[2] >= 0.0          # opposition surge amplitude
        assert lo[3] > 0.0           # surge width
        assert lo[4] >= 0.0 and hi[4] <= 90.0   # mean slope angle
