"""Test TModel class.

Tests the init, grow_ttree and other methods of TModel.
"""

from importlib import resources

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

# Fixtures: inputs and expected values from the original implementation in R


@pytest.fixture(scope="module")
def rvalues():
    """Fixture to load test inputs from file.

    The regression test inputs consist of time series of growth from an initial DBH
    using a small set of different plant functional type definitions.
    """
    from pyrealm.demography.flora import PlantFunctionalType

    # Load the PFT definitions and rename to pyrealm attributes
    pfts_path = resources.files("pyrealm_build_data.t_model") / "pft_definitions.csv"
    pft_definitions = pd.read_csv(pfts_path)

    # Map the PFT trait args from the R implementation to pyrealm
    pft_definitions = pft_definitions.rename(
        columns={
            "a": "a_hd",
            "cr": "ca_ratio",
            "Hm": "h_max",
            "rho": "rho_s",
            "L": "lai",
            "sigma": "sla",
            "tf": "tau_f",
            "tr": "tau_r",
            "K": "par_ext",
            "y": "yld",
            "rr": "resp_r",
            "rs": "resp_s",
        }
    )

    # Convert to dicts for creating PFT instances
    pft_definitions = pft_definitions.to_dict(orient="records")

    return_value = []

    # Loop over the PFT definitions
    for pft_args in pft_definitions:
        # Record the starting DBH and create the PFT instances
        dbh_init = pft_args.pop("d")
        pft = PlantFunctionalType(**pft_args)

        # Load the appropriate output file and then remap the field names
        datapath = (
            resources.files("pyrealm_build_data.t_model")
            / f"rtmodel_output_{pft.name}.csv"
        )
        data = pd.read_csv(datapath)

        data = data.rename(
            columns={
                "dD": "delta_d",
                "D": "diameter",
                "H": "height",
                "fc": "crown_fraction",
                "Ac": "crown_area",
                "Wf": "mass_fol",
                "Ws": "mass_stm",
                "Wss": "mass_swd",
                "GPP": "gpp_actual",
                "Rm1": "resp_swd",
                "Rm2": "resp_frt",
                "dWs": "delta_mass_stm",
                "dWfr": "delta_mass_frt",
            }
        )

        # Fix some scaling differences:
        # The R tmodel implementation rescales reported delta_d as a radial increase in
        # millimetres, not diameter increase in metres
        data["delta_d"] = data["delta_d"] / 500

        # The R tmodel implementation slices off foliar respiration costs from GPP
        # before doing anything - the pyrealm.tmodel implementation keeps this cost
        # within the tree calculation, so proportionally inflate the GPP to make it
        # match
        data["gpp_actual"] = data["gpp_actual"] / (1 - pft.resp_f)

        # Add a tuple of the inputs and outputs to the return list.
        return_value.append((pft, dbh_init, data))

    return return_value


def test_calculate_heights(rvalues):
    """Test calculation of heights of tree from diameter."""

    from pyrealm.demography.t_model_functions import calculate_heights

    for pft, _, data in rvalues:
        actual_heights = calculate_heights(
            h_max=pft.h_max, a_hd=pft.a_hd, dbh=data["diameter"]
        )

        assert_array_almost_equal(actual_heights, data["height"], decimal=8)


def test_calculate_crown_areas(rvalues):
    """Tests calculation of crown areas of trees."""
    from pyrealm.demography.t_model_functions import calculate_crown_areas

    for pft, _, data in rvalues:
        actual_crown_areas = calculate_crown_areas(
            ca_ratio=pft.ca_ratio,
            a_hd=pft.a_hd,
            dbh=data["diameter"],
            height=data["height"],
        )

        assert_array_almost_equal(actual_crown_areas, data["crown_area"], decimal=8)


def test_calculate_crown_fractions(rvalues):
    """Tests calculation of crown fractions of trees."""

    from pyrealm.demography.t_model_functions import calculate_crown_fractions

    for pft, _, data in rvalues:
        actual_crown_fractions = calculate_crown_fractions(
            a_hd=pft.a_hd,
            dbh=data["diameter"],
            height=data["height"],
        )
        assert_array_almost_equal(
            actual_crown_fractions, data["crown_fraction"], decimal=8
        )


def test_calculate_stem_masses(rvalues):
    """Tests happy path for calculation of stem masses of trees."""

    from pyrealm.demography.t_model_functions import calculate_stem_masses

    for pft, _, data in rvalues:
        actual_stem_masses = calculate_stem_masses(
            dbh=data["diameter"],
            height=data["height"],
            rho_s=pft.rho_s,
        )
        assert_array_almost_equal(actual_stem_masses, data["mass_stm"], decimal=8)


def test_calculate_foliage_masses(rvalues):
    """Tests calculation of foliage masses of trees."""

    from pyrealm.demography.t_model_functions import calculate_foliage_masses

    for pft, _, data in rvalues:
        actual_foliage_masses = calculate_foliage_masses(
            crown_area=data["crown_area"], lai=pft.lai, sla=pft.sla
        )
        assert_array_almost_equal(actual_foliage_masses, data["mass_fol"], decimal=8)


def test_calculate_sapwood_masses(rvalues):
    """Tests calculation of sapwood masses of trees."""

    from pyrealm.demography.t_model_functions import calculate_sapwood_masses

    for pft, _, data in rvalues:
        actual_sapwood_masses = calculate_sapwood_masses(
            crown_area=data["crown_area"],
            height=data["height"],
            crown_fraction=data["crown_fraction"],
            ca_ratio=pft.ca_ratio,
            rho_s=pft.rho_s,
        )
        assert_array_almost_equal(actual_sapwood_masses, data["mass_swd"], decimal=8)
