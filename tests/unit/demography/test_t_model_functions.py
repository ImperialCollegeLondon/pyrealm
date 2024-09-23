"""test the functions in t_model_functions.py."""

from contextlib import nullcontext as does_not_raise
from importlib import resources

import numpy as np
import pandas as pd
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


@pytest.fixture
def rtmodel_data():
    """Loads some simple predictions from the R implementation for testing."""

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

    # Add foliar respiration rate as 0.1, as this is handled outside of the R
    # implementation as a function of GPP.
    pft_definitions["resp_f"] = 0.1

    rdata_path = (
        resources.files("pyrealm_build_data.t_model") / "rtmodel_unit_testing.csv"
    )
    rdata = pd.read_csv(rdata_path)

    rdata = rdata.rename(
        columns={
            "dD": "delta_d",
            "D": "diameter",
            "H": "height",
            "fc": "crown_fraction",
            "Ac": "crown_area",
            "Wf": "mass_fol",
            "Ws": "mass_stm",
            "Wss": "mass_swd",
            "P0": "potential_gpp",
            "GPP": "crown_gpp",
            "Rm1": "resp_swd",
            "Rm2": "resp_frt",
            "dWs": "delta_mass_stm",
            "dWfr": "delta_mass_frt",
        }
    )

    # Fix some scaling differences:
    # The R tmodel implementation rescales reported delta_d as a radial increase in
    # millimetres, not diameter increase in metres
    rdata["delta_d"] = rdata["delta_d"] / 500

    # Wrap the return data into arrays with PFT as columns and diameter values as rows
    pft_arrays = {k: v.to_numpy() for k, v in pft_definitions.items()}
    rdata_arrays = {k: np.reshape(v, (3, 6)).T for k, v in rdata.items()}

    return pft_arrays, rdata_arrays


@pytest.mark.parametrize(
    argnames="data_idx, pft_idx, out_idx, exp_shape",
    argvalues=[
        pytest.param((0, slice(None)), slice(None), (0, slice(None)), (3,), id="row_0"),
        pytest.param((1, slice(None)), slice(None), (1, slice(None)), (3,), id="row_1"),
        pytest.param((2, slice(None)), slice(None), (2, slice(None)), (3,), id="row_2"),
        pytest.param((3, slice(None)), slice(None), (3, slice(None)), (3,), id="row_3"),
        pytest.param((4, slice(None)), slice(None), (4, slice(None)), (3,), id="row_4"),
        pytest.param((5, slice(None)), slice(None), (5, slice(None)), (3,), id="row_5"),
        pytest.param((slice(None), [0]), [0], (slice(None), [0]), (6, 1), id="col_0"),
        pytest.param((slice(None), [1]), [1], (slice(None), [1]), (6, 1), id="col_1"),
        pytest.param((slice(None), [2]), [2], (slice(None), [2]), (6, 1), id="col_2"),
        pytest.param(
            (slice(None), [0]),
            [0, 0, 0],
            (slice(None), [0, 0, 0]),
            (6, 3),
            id="column_broadcast_0",
        ),
        pytest.param(
            (slice(None), [1]),
            [1, 1, 1],
            (slice(None), [1, 1, 1]),
            (6, 3),
            id="column_broadcast_1",
        ),
        pytest.param(
            (slice(None), [2]),
            [2, 2, 2],
            (slice(None), [2, 2, 2]),
            (6, 3),
            id="column_broadcast_2",
        ),
        pytest.param(
            (slice(None), slice(None)),
            slice(None),
            (slice(None), slice(None)),
            (6, 3),
            id="array",
        ),
    ],
)
class TestTModel:
    """Test T Model functions.

    A class is used here to pass a shared parameterisation of input array shapes to each
    of the T model functions. The combination of data_idx and pft_idx slice up the
    inputs to provide a wide range of input shape combinations

    * each data row against full pft array: (3,) + (3,) -> (3,)
    * each column against a scalar pft array for a single PFT: (6,1) + (1,) -> (6,1)
    * each column broadcast against a row array of three PFTs: (6,1) + (3,) -> (6,3)
    * whole array against full pft array: (6,3) + (3,) -> (6,3)

    The column broadcast has an added complexity, which is that the data values in the
    columns are PFT specific predictions (apart from the initial stem diameters), so do
    not match if a single column is broadcast across PFTs. To get around this and test
    the broadcasting, these tests duplicate a single PFT trait to (3,) and duplicate the
    expected outputs to repeat the single column expectations across (6, 3)
    """

    def test_calculate_heights(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of heights of tree from diameter."""
        from pyrealm.demography.t_model_functions import calculate_heights

        pfts, data = rtmodel_data

        result = calculate_heights(
            h_max=pfts["h_max"][pft_idx],
            a_hd=pfts["a_hd"][pft_idx],
            dbh=data["diameter"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["height"][out_idx])

    def test_calculate_crown_areas(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of crown areas of trees."""

        from pyrealm.demography.t_model_functions import calculate_crown_areas

        pfts, data = rtmodel_data

        result = calculate_crown_areas(
            ca_ratio=pfts["ca_ratio"][pft_idx],
            a_hd=pfts["a_hd"][pft_idx],
            dbh=data["diameter"][data_idx],
            stem_height=data["height"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["crown_area"][out_idx])

    def test_calculate_crown_fractions(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of crown fraction of trees."""

        from pyrealm.demography.t_model_functions import calculate_crown_fractions

        pfts, data = rtmodel_data

        result = calculate_crown_fractions(
            a_hd=pfts["a_hd"][pft_idx],
            dbh=data["diameter"][data_idx],
            stem_height=data["height"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["crown_fraction"][out_idx])

    def test_calculate_stem_masses(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography.t_model_functions import calculate_stem_masses

        pfts, data = rtmodel_data

        result = calculate_stem_masses(
            rho_s=pfts["rho_s"][pft_idx],
            dbh=data["diameter"][data_idx],
            stem_height=data["height"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["mass_stm"][out_idx])

    def test_calculate_foliage_masses(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography.t_model_functions import calculate_foliage_masses

        pfts, data = rtmodel_data

        result = calculate_foliage_masses(
            lai=pfts["lai"][pft_idx],
            sla=pfts["sla"][pft_idx],
            crown_area=data["crown_area"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["mass_fol"][out_idx])

    def test_calculate_sapwood_masses(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography.t_model_functions import calculate_sapwood_masses

        pfts, data = rtmodel_data

        result = calculate_sapwood_masses(
            rho_s=pfts["rho_s"][pft_idx],
            ca_ratio=pfts["ca_ratio"][pft_idx],
            crown_area=data["crown_area"][data_idx],
            stem_height=data["height"][data_idx],
            crown_fraction=data["crown_fraction"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["mass_swd"][out_idx])

    def test_calculate_whole_crown_gpp(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of whole crown GPP."""

        from pyrealm.demography.t_model_functions import calculate_whole_crown_gpp

        pfts, data = rtmodel_data

        result = calculate_whole_crown_gpp(
            lai=pfts["lai"][pft_idx],
            par_ext=pfts["par_ext"][pft_idx],
            crown_area=data["crown_area"][data_idx],
            potential_gpp=data["potential_gpp"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["crown_gpp"][out_idx])

    def test_calculate_sapwood_respiration(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of sapwood respiration."""

        from pyrealm.demography.t_model_functions import calculate_sapwood_respiration

        pfts, data = rtmodel_data

        result = calculate_sapwood_respiration(
            resp_s=pfts["resp_s"][pft_idx],
            sapwood_mass=data["mass_swd"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["resp_swd"][out_idx])

    def test_calculate_foliar_respiration(
        self, request, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of foliar respiration.

        NOTE - this test is extremely circular, because the R implementation does not
        apply this in the same way. This is mostly just to validate array shape and
        broadcasting
        """

        from pyrealm.demography.t_model_functions import calculate_foliar_respiration

        pfts, data = rtmodel_data

        result = calculate_foliar_respiration(
            resp_f=pfts["resp_f"][pft_idx],
            whole_crown_gpp=data["crown_gpp"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(
            result, data["crown_gpp"][data_idx] * pfts["resp_f"][pft_idx]
        )

    def test_calculate_fine_root_respiration(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of fine root respiration."""

        from pyrealm.demography.t_model_functions import calculate_fine_root_respiration

        pfts, data = rtmodel_data

        result = calculate_fine_root_respiration(
            zeta=pfts["zeta"][pft_idx],
            sla=pfts["sla"][pft_idx],
            resp_r=pfts["resp_r"][pft_idx],
            foliage_mass=data["mass_fol"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["resp_frt"][out_idx])

    def test_calculate_net_primary_productivity(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of net primary productivity."""

        from pyrealm.demography.t_model_functions import (
            calculate_net_primary_productivity,
        )

        pfts, data = rtmodel_data

        result = calculate_net_primary_productivity(
            yld=pfts["yld"][pft_idx],
            whole_crown_gpp=data["crown_gpp"][data_idx],
            foliar_respiration=0,  # Not included here in the R implementation
            fine_root_respiration=data["resp_frt"][data_idx],
            sapwood_respiration=data["resp_swd"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["NPP"][out_idx])

    def test_calculate_foliage_and_fine_root_turnover(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of foliage and fine root turnover."""

        from pyrealm.demography.t_model_functions import (
            calculate_foliage_and_fine_root_turnover,
        )

        pfts, data = rtmodel_data

        result = calculate_foliage_and_fine_root_turnover(
            sla=pfts["sla"][pft_idx],
            zeta=pfts["zeta"][pft_idx],
            tau_f=pfts["tau_f"][pft_idx],
            tau_r=pfts["tau_r"][pft_idx],
            foliage_mass=data["mass_fol"][data_idx],
        )

        assert result.shape == exp_shape
        assert np.allclose(result, data["turnover"][out_idx])

    def test_calculate_calculate_growth_increments(
        self, rtmodel_data, data_idx, pft_idx, out_idx, exp_shape
    ):
        """Tests calculation of growth increments."""

        from pyrealm.demography.t_model_functions import (
            calculate_growth_increments,
        )

        pfts, data = rtmodel_data

        delta_d, delta_mass_stm, delta_mass_frt = calculate_growth_increments(
            rho_s=pfts["rho_s"][pft_idx],
            a_hd=pfts["a_hd"][pft_idx],
            h_max=pfts["h_max"][pft_idx],
            lai=pfts["lai"][pft_idx],
            ca_ratio=pfts["ca_ratio"][pft_idx],
            sla=pfts["sla"][pft_idx],
            zeta=pfts["zeta"][pft_idx],
            npp=data["NPP"][data_idx],
            turnover=data["turnover"][data_idx],
            dbh=data["diameter"][data_idx],
            stem_height=data["height"][data_idx],
        )

        assert delta_d.shape == exp_shape
        assert np.allclose(delta_d, data["delta_d"][out_idx])

        assert delta_mass_stm.shape == exp_shape
        assert np.allclose(delta_mass_stm, data["delta_mass_stm"][out_idx])

        assert delta_mass_frt.shape == exp_shape
        assert np.allclose(delta_mass_frt, data["delta_mass_frt"][out_idx])
