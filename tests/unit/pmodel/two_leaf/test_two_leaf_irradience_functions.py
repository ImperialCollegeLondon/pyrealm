"""test_two_leaf_functions.

This module contains pytest unit tests for validating the correctness of various
functions involved in the calculation of photosynthetic parameters and canopy
irradiance models. The functions tested include those related to beam extinction
coefficients, irradiance calculations, carboxylation scaling, electron transport
rates, and gross primary productivity (GPP).

Each test function is parameterized with a variety of input scenarios to ensure
robustness and accuracy of the implemented algorithms.

Tests are designed to cover edge cases, typical use cases, and mixed scenarios,
providing comprehensive coverage of the functionality.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

# from pyrealm.pmodel.two_leaf import (
#     Jmax25,
#     Jmax25_temp_correction,
#     Vmax25_canopy,
#     Vmax25_shade,
#     Vmax25_sun,
#     assimilation_canopy,
#     assimilation_rate,
#     beam_extinction_coeff,
#     beam_irrad_unif_leaf_angle_dist,
#     beam_irradience,
#     beam_irradience_h_leaves,
#     canopy_extinction_coefficient,
#     canopy_irradience,
#     carboxylation_scaling_to_T,
#     diffuse_radiation,
#     electron_transport_rate,
#     fraction_of_diffuse_rad,
#     gross_primary_product,
#     photosynthetic_estimate,
#     scattered_beam_extinction_coeff,
#     scattered_beam_irradience,
#     shaded_absorbed_irrad,
#     sunlit_absorbed_irrad,
#     sunlit_beam_irrad,
#     sunlit_diffuse_irrad,
#     sunlit_scattered_irrad,
# )


@pytest.mark.parametrize(
    "k_sigma, expected_rho_h",
    [
        # Test case 1: Typical value for k_sigma
        (0.5, 0.1715728752538099),
    ],
)
def test_beam_irradience_h_leaves(k_sigma, expected_rho_h):
    """Test that TwoLeafConstants.

    Check it correctly calculates horizontal_leaf_reflectance (rho_h) from the
    leaf_scattering_coef (sigma).
    """

    from pyrealm.constants.two_leaf import TwoLeafConst

    const = TwoLeafConst(leaf_scattering_coef=0.5)

    assert_array_almost_equal(
        const.horizontal_leaf_reflectance, expected_rho_h, decimal=5
    )


@pytest.mark.parametrize(
    "args, expected_kb",
    [
        pytest.param(
            {"solar_elevation": np.array([0.6, 0.7])},
            np.array([0.5 / np.sin(0.6), 0.5 / np.sin(0.7)]),
            id="defaults: beta_angle > k_sol_obs_angle",
        ),
        pytest.param(
            {"solar_elevation": np.array([0.01, 0.009])},
            np.repeat([0.5 / np.sin(np.pi / 180)], 2),
            id="defaults: beta_angle < k_sol_obs_angle",
        ),
        pytest.param(
            {"solar_elevation": np.array([0.01, 0.7])},
            np.array([0.5 / np.sin(np.pi / 180), 0.5 / np.sin(0.7)]),
            id="defaults: mixed",
        ),
        pytest.param(
            {
                "solar_elevation": np.array([0.01, 0.7]),
                "solar_obscurity_angle": np.pi / 90,
                "extinction_numerator": 0.6,
            },
            np.array([0.6 / np.sin(np.pi / 90), 0.6 / np.sin(0.7)]),
            id="custom",
        ),
    ],
)
def test_beam_extinction_coeff(args, expected_kb):
    """Test the beam_extinction_coeff function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_beam_extinction_coef

    result = calculate_beam_extinction_coef(**args)

    assert_allclose(result, expected_kb)


@pytest.mark.parametrize(
    "patm, beta_angle, expected_fd",
    [
        # Test case 1: Basic case with standard pressure and moderate angles
        (
            np.array([101325, 101300]),
            np.array([0.6, 0.7]),
            np.array(
                [
                    (1 - 0.72 ** ((101325 / 101325) / np.sin(0.6)))
                    / (
                        1
                        + (0.72 ** ((101325 / 101325) / np.sin(0.6)) * (1 / 0.426 - 1))
                    ),
                    (1 - 0.72 ** ((101300 / 101325) / np.sin(0.7)))
                    / (
                        1
                        + (0.72 ** ((101300 / 101325) / np.sin(0.7)) * (1 / 0.426 - 1))
                    ),
                ]
            ),
        ),
    ],
)
def test_fraction_of_diffuse_rad(patm, beta_angle, expected_fd):
    """Test the fraction_of_diffuse_rad function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_fraction_of_diffuse_radiation

    result = calculate_fraction_of_diffuse_radiation(
        patm=patm, solar_elevation=beta_angle
    )
    assert_allclose(result, expected_fd)


@pytest.mark.parametrize(
    "rho_h, kb, expected_rho_cb",
    [
        # Test case 1: Typical values for rho_h and kb
        (
            np.array([0.1, 0.2]),
            np.array([0.5, 1.0]),
            np.array([0.064493015, 0.181269247]),
        ),
    ],
)
def test_beam_irrad_unif_leaf_angle_dist(rho_h, kb, expected_rho_cb):
    """Test beam_irrad_unif_leaf_angle_dist function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_beam_reflectance

    result = calculate_beam_reflectance(
        beam_extinction=kb, horizontal_leaf_reflectance=rho_h
    )
    assert_allclose(result, expected_rho_cb)


@pytest.mark.parametrize(
    "rho_cb, I_b, kb_prime, I_d, leaf_area_index,  expected_I_c",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([0.2, 0.3]),
            np.array([500.0, 300.0]),
            np.array([0.6, 0.4]),
            np.array([200.0, 150.0]),
            np.array([2.0, 1.5]),
            np.array([426.55119, 190.17033]),
        ),
    ],
)
def test_canopy_irradience(rho_cb, I_b, kb_prime, I_d, leaf_area_index, expected_I_c):
    """Test the canopy_irradience function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_canopy_irradiance

    result = calculate_canopy_irradiance(
        beam_reflectance=rho_cb,
        beam_irradiance=I_b,
        scattered_beam_extinction_coef=kb_prime,
        diffuse_radiation=I_d,
        leaf_area_index=leaf_area_index,
    )
    assert_allclose(result, expected_I_c)


@pytest.mark.parametrize(
    "I_b, kb, leaf_area_index, expected_Isun_beam",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([500.0, 300.0]),
            np.array([0.7, 0.5]),
            np.array([2.0, 1.5]),
            np.array([320.19629, 134.546529]),
        ),
    ],
)
def test_sunlit_beam_irrad(I_b, kb, leaf_area_index, expected_Isun_beam):
    """Test the sunlit_beam_irrad function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_sunlit_beam_irradiance

    result = calculate_sunlit_beam_irradiance(
        beam_irradiance=I_b, beam_extinction_coef=kb, leaf_area_index=leaf_area_index
    )
    assert_allclose(result, expected_Isun_beam)


@pytest.mark.parametrize(
    "I_d, kb, leaf_area_index, expected_Isun_diffuse",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([200.0, 150.0]),
            np.array([0.7, 0.5]),
            np.array([2.0, 1.5]),
            np.array([91.97169, 71.58702]),
        ),
    ],
)
def test_sunlit_diffuse_irrad(I_d, kb, leaf_area_index, expected_Isun_diffuse):
    """Test the sunlit_diffuse_irrad function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_sunlit_diffuse_irradiance

    result = calculate_sunlit_diffuse_irradiance(
        diffuse_irradiance=I_d, beam_extinction_coef=kb, leaf_area_index=leaf_area_index
    )

    assert_allclose(result, expected_Isun_diffuse)


@pytest.mark.parametrize(
    "I_b, rho_cb, kb_prime, kb, leaf_area_index, expected_Isun_scattered",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([500.0, 300.0]),
            np.array([0.2, 0.3]),
            np.array([0.6, 0.5]),
            np.array([0.7, 0.6]),
            np.array([2.0, 1.5]),
            np.array([-28.674522, -29.301883]),
        ),
    ],
)
def test_sunlit_scattered_irrad(
    I_b, rho_cb, kb_prime, kb, leaf_area_index, expected_Isun_scattered
):
    """Test the sunlit_scattered_irrad function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_sunlit_scattered_irradiance

    result = calculate_sunlit_scattered_irradiance(
        beam_irradiance=I_b,
        beam_reflectance=rho_cb,
        scattered_beam_extinction_coef=kb_prime,
        beam_extinction_coef=kb,
        leaf_area_index=leaf_area_index,
    )
    assert_allclose(result, expected_Isun_scattered)


@pytest.mark.parametrize(
    "vcmax_pmod, expected_kv_Lloyd",
    [
        # Test case 1: Typical value for vcmax_pmod
        (np.array([50.0, 60.0]), np.array([0.142487643, 0.156891625])),
    ],
)
def test_canopy_extinction_coefficient(vcmax_pmod, expected_kv_Lloyd):
    """Test the canopy_extinction_coefficient function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_canopy_extinction_coef

    result = calculate_canopy_extinction_coef(vcmax_pmod)
    assert_allclose(result, expected_kv_Lloyd)


@pytest.mark.parametrize(
    "leaf_area_index, vcmax25_pmod, kv, expected_Vmax25_canopy",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([2.0, 3.0]),
            np.array([50.0, 60.0]),
            np.array([0.5, 0.6]),
            np.array([78.69386806, 135.3565092]),
        ),
    ],
)
def test_Vmax25_canopy(leaf_area_index, vcmax25_pmod, kv, expected_Vmax25_canopy):
    """Test the Vmax25_canopy function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_canopy_vcmax25

    result = calculate_canopy_vcmax25(leaf_area_index, vcmax25_pmod, kv)
    assert_allclose(result, expected_Vmax25_canopy)


@pytest.mark.parametrize(
    "leaf_area_index, vcmax25_pmod, kv, kb, expected_Vmax25_sun",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([2.0, 3.0]),
            np.array([50.0, 60.0]),
            np.array([0.5, 0.6]),
            np.array([0.7, 0.8]),
            np.array([44.75954636, 57.0127759]),
        ),
    ],
)
def test_Vmax25_sun(leaf_area_index, vcmax25_pmod, kv, kb, expected_Vmax25_sun):
    """Test the Vmax25_sun function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_sun_vcmax25

    result = calculate_sun_vcmax25(leaf_area_index, vcmax25_pmod, kv, kb)
    assert_allclose(result, expected_Vmax25_sun)


@pytest.mark.parametrize(
    "Jmax, I_c, expected_J",
    [
        # Test case 1: Typical values for Jmax and I_c
        (
            np.array([100.0, 150.0]),
            np.array([500.0, 400.0]),
            np.array([59.02777778, 69.8630137]),
        ),
    ],
)
def test_electron_transport_rate(Jmax, I_c, expected_J):
    """Test the electron_transport_rate function with various input scenarios."""

    from pyrealm.pmodel.two_leaf import calculate_electron_transport_rate

    result = calculate_electron_transport_rate(jmax=Jmax, absorbed_irradiance=I_c)
    assert_allclose(result, expected_J)
