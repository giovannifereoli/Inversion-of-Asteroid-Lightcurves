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
