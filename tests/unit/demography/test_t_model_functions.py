"""test the functions in t_model_functions.py."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="pft_args, size_args, outcome, excep_message",
    argvalues=[
        pytest.param(
            [np.ones(4), np.ones(4)],
            [np.ones(4), np.ones(4)],
            does_not_raise(),
            None,
            id="all_1d_ok",
        ),
        pytest.param(
            [np.ones(5), np.ones(4)],
            [np.ones(4), np.ones(4)],
            pytest.raises(ValueError),
            "PFT trait values are not of equal length",
            id="pfts_unequal",
        ),
        pytest.param(
            [np.ones(4), np.ones(4)],
            [np.ones(5), np.ones(4)],
            pytest.raises(ValueError),
            "Size arrays are not of equal length",
            id="shape_unequal",
        ),
        pytest.param(
            [np.ones((4, 2)), np.ones((4, 2))],
            [np.ones(4), np.ones(4)],
            pytest.raises(ValueError),
            "T model functions only accept 1D arrays of PFT trait values",
            id="pfts_not_row_arrays",
        ),
        pytest.param(
            [np.ones(4), np.ones(4)],
            [np.ones(5), np.ones(5)],
            pytest.raises(ValueError),
            "PFT and size inputs to T model function are not compatible.",
            id="sizes_row_array_of_bad_length",
        ),
        pytest.param(
            [np.ones(4), np.ones(4)],
            [np.ones((5, 1)), np.ones((5, 1))],
            does_not_raise(),
            None,
            id="size_2d_columns_ok",
        ),
        pytest.param(
            [np.ones(4), np.ones(4)],
            [np.ones((5, 2)), np.ones((5, 2))],
            pytest.raises(ValueError),
            "PFT and size inputs to T model function are not compatible.",
            id="size_2d_not_ok",
        ),
        pytest.param(
            [np.ones(4), np.ones(4)],
            [np.ones((5, 4)), np.ones((5, 4))],
            does_not_raise(),
            None,
            id="size_2d_weird_but_ok",
        ),
    ],
)
def test__validate_t_model_args(pft_args, size_args, outcome, excep_message):
    """Test shared input validation function."""
    from pyrealm.demography.t_model_functions import _validate_t_model_args

    with outcome as excep:
        _validate_t_model_args(pft_args=pft_args, size_args=size_args)
        return

    assert str(excep.value).startswith(excep_message)


def test_calculate_heights():
    """Tests happy path for calculation of heights of tree from diameter."""

    from pyrealm.demography.t_model_functions import calculate_heights

    pft_h_max_values = np.array([25.33, 15.33])
    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    expected_heights = np.array([15.19414157, 15.16639589])
    actual_heights = calculate_heights(
        h_max=pft_h_max_values,
        a_hd=pft_a_hd_values,
        dbh=diameters_at_breast_height,
    )

    np.allclose(actual_heights, expected_heights)


def test_calculate_crown_areas():
    """Tests happy path for calculation of crown areas of trees."""

    from pyrealm.demography.t_model_functions import calculate_crown_areas

    pft_ca_ratio_values = np.array([2, 3])
    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    stem_height = np.array([15.194142, 15.166396])
    expected_crown_areas = np.array([0.04114983, 0.1848361])
    actual_crown_areas = calculate_crown_areas(
        ca_ratio=pft_ca_ratio_values,
        a_hd=pft_a_hd_values,
        dbh=diameters_at_breast_height,
        stem_height=stem_height,
    )

    np.allclose(actual_crown_areas, expected_crown_areas)


def test_calculate_crown_fractions():
    """Tests happy path for calculation of crown fractions of trees."""

    from pyrealm.demography.t_model_functions import calculate_crown_fractions

    pft_a_hd_values = np.array([116.0, 116.0])
    diameters_at_breast_height = np.array([0.2, 0.6])
    stem_height = np.array([15.194142, 15.166396])
    expected_crown_fractions = np.array([0.65491991, 0.21790799])
    actual_crown_fractions = calculate_crown_fractions(
        a_hd=pft_a_hd_values,
        dbh=diameters_at_breast_height,
        stem_height=stem_height,
    )

    np.allclose(actual_crown_fractions, expected_crown_fractions)


def test_calculate_stem_masses():
    """Tests happy path for calculation of stem masses."""

    from pyrealm.demography.t_model_functions import calculate_stem_masses

    diameters_at_breast_height = np.array([0.2, 0.6])
    stem_height = np.array([15.194142, 15.166396])
    pft_rho_s_values = np.array([200.0, 200.0])
    expected_stem_masses = np.array([47.73380488, 428.8197443])
    actual_stem_masses = calculate_stem_masses(
        dbh=diameters_at_breast_height, stem_height=stem_height, rho_s=pft_rho_s_values
    )

    np.allclose(actual_stem_masses, expected_stem_masses)


def test_calculate_foliage_masses():
    """Tests happy path for calculation of foliage masses."""

    from pyrealm.demography.t_model_functions import calculate_foliage_masses

    crown_areas = np.array([0.04114983, 0.1848361])
    pft_lai_values = np.array([1.8, 1.8])
    pft_sla_values = np.array([14.0, 14.0])
    expected_foliage_masses = np.array([0.00529069, 0.02376464])
    actual_foliage_masses = calculate_foliage_masses(
        crown_area=crown_areas, lai=pft_lai_values, sla=pft_sla_values
    )

    np.allclose(actual_foliage_masses, expected_foliage_masses)


def test_calculate_sapwood_masses():
    """Tests happy path for calculation of sapwood masses."""

    from pyrealm.demography.t_model_functions import calculate_sapwood_masses

    crown_areas = np.array([0.04114983, 0.1848361])
    pft_rho_s_values = np.array([200.0, 200.0])
    stem_height = np.array([15.194142, 15.166396])
    crown_fractions = np.array([0.65491991, 0.21790799])
    pft_ca_ratio_values = [390.43, 390.43]
    expected_sapwood_masses = np.array([0.21540173, 1.27954667])
    actual_sapwood_masses = calculate_sapwood_masses(
        crown_area=crown_areas,
        rho_s=pft_rho_s_values,
        stem_height=stem_height,
        crown_fraction=crown_fractions,
        ca_ratio=pft_ca_ratio_values,
    )

    np.allclose(actual_sapwood_masses, expected_sapwood_masses)
