"""test the functions in t_model_functions.py."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="crown_areas, expected_r0",
    argvalues=(
        (np.array([20, 30]), np.array([0.86887756, 1.29007041])),
        (np.array([30, 40]), np.array([1.06415334, 1.489645])),
    ),
)
def test_calculate_crown_r_0_values(crown_areas, expected_r0):
    """Test happy path for calculating r_0."""

    from pyrealm.demography.t_model_functions import calculate_crown_r0

    q_m = np.array([2.9038988210485766, 2.3953681843215673])
    actual_r0_values = calculate_crown_r0(q_m=q_m, crown_area=crown_areas)

    assert np.allclose(actual_r0_values, expected_r0)


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
            "Trait and size inputs are row arrays of unequal length.",
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


@pytest.mark.parametrize(
    argnames="data_idx, pft_idx, outcome, excep_msg, out_idx, exp_shape",
    argvalues=[
        pytest.param(
            (0, slice(None)),
            slice(None),
            does_not_raise(),
            None,
            (0, slice(None)),
            (3,),
            id="row_0",
        ),
        pytest.param(
            (1, slice(None)),
            slice(None),
            does_not_raise(),
            None,
            (1, slice(None)),
            (3,),
            id="row_1",
        ),
        pytest.param(
            (2, slice(None)),
            slice(None),
            does_not_raise(),
            None,
            (2, slice(None)),
            (3,),
            id="row_2",
        ),
        pytest.param(
            (3, slice(None)),
            slice(None),
            does_not_raise(),
            None,
            (3, slice(None)),
            (3,),
            id="row_3",
        ),
        pytest.param(
            (4, slice(None)),
            slice(None),
            does_not_raise(),
            None,
            (4, slice(None)),
            (3,),
            id="row_4",
        ),
        pytest.param(
            (5, slice(None)),
            slice(None),
            does_not_raise(),
            None,
            (5, slice(None)),
            (3,),
            id="row_5",
        ),
        pytest.param(
            (slice(None), [0]),
            [0],
            does_not_raise(),
            None,
            (slice(None), [0]),
            (6, 1),
            id="col_0",
        ),
        pytest.param(
            (slice(None), [1]),
            [1],
            does_not_raise(),
            None,
            (slice(None), [1]),
            (6, 1),
            id="col_1",
        ),
        pytest.param(
            (slice(None), [2]),
            [2],
            does_not_raise(),
            None,
            (slice(None), [2]),
            (6, 1),
            id="col_2",
        ),
        pytest.param(
            (slice(None), [0]),
            [0, 0, 0],
            does_not_raise(),
            None,
            (slice(None), [0, 0, 0]),
            (6, 3),
            id="column_broadcast_0",
        ),
        pytest.param(
            (slice(None), [1]),
            [1, 1, 1],
            does_not_raise(),
            None,
            (slice(None), [1, 1, 1]),
            (6, 3),
            id="column_broadcast_1",
        ),
        pytest.param(
            (slice(None), [2]),
            [2, 2, 2],
            does_not_raise(),
            None,
            (slice(None), [2, 2, 2]),
            (6, 3),
            id="column_broadcast_2",
        ),
        pytest.param(
            (slice(None), slice(None)),
            slice(None),
            does_not_raise(),
            None,
            (slice(None), slice(None)),
            (6, 3),
            id="array",
        ),
        pytest.param(
            (0, slice(None)),
            [0, 1, 2, 0],
            pytest.raises(ValueError),
            "Trait and size inputs are row arrays of unequal length.",
            None,
            None,
            id="fail_PFT_and_sizes_rows_but_not_equal_length",
        ),
        pytest.param(
            (0, slice(None)),
            np.newaxis,
            pytest.raises(ValueError),
            "T model functions only accept 1D arrays of PFT trait values",
            None,
            None,
            id="fail_2D_PFT",
        ),
        pytest.param(
            (slice(None), [0, 1]),
            slice(None),
            pytest.raises(ValueError),
            "PFT and size inputs to T model function are not compatible.",
            None,
            None,
            id="fail_badly_shaped_2D",
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
    expected outputs to repeat the single column expectations across (6, 3).

    The parameterization also includes three cases that check the failure modes for
    inputs.
    """

    def test_calculate_heights(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of heights of tree from diameter."""
        from pyrealm.demography.t_model_functions import calculate_heights

        with outcome as excep:
            result = calculate_heights(
                h_max=rtmodel_flora.h_max[pft_idx],
                a_hd=rtmodel_flora.a_hd[pft_idx],
                dbh=rtmodel_data["dbh"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["stem_height"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_dbh_from_height(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests inverted calculation of dbh from height."""

        from pyrealm.demography.t_model_functions import calculate_dbh_from_height

        with outcome as excep:
            result = calculate_dbh_from_height(
                h_max=rtmodel_flora.h_max[pft_idx],
                a_hd=rtmodel_flora.a_hd[pft_idx],
                stem_height=rtmodel_data["stem_height"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["dbh"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_crown_areas(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of crown areas of trees."""

        from pyrealm.demography.t_model_functions import calculate_crown_areas

        with outcome as excep:
            result = calculate_crown_areas(
                ca_ratio=rtmodel_flora.ca_ratio[pft_idx],
                a_hd=rtmodel_flora.a_hd[pft_idx],
                dbh=rtmodel_data["dbh"][data_idx],
                stem_height=rtmodel_data["stem_height"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["crown_area"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_crown_fractions(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of crown fraction of trees."""

        from pyrealm.demography.t_model_functions import calculate_crown_fractions

        with outcome as excep:
            result = calculate_crown_fractions(
                a_hd=rtmodel_flora.a_hd[pft_idx],
                dbh=rtmodel_data["dbh"][data_idx],
                stem_height=rtmodel_data["stem_height"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["crown_fraction"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_stem_masses(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography.t_model_functions import calculate_stem_masses

        with outcome as excep:
            result = calculate_stem_masses(
                rho_s=rtmodel_flora.rho_s[pft_idx],
                dbh=rtmodel_data["dbh"][data_idx],
                stem_height=rtmodel_data["stem_height"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["stem_mass"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_foliage_masses(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography.t_model_functions import calculate_foliage_masses

        with outcome as excep:
            result = calculate_foliage_masses(
                lai=rtmodel_flora.lai[pft_idx],
                sla=rtmodel_flora.sla[pft_idx],
                crown_area=rtmodel_data["crown_area"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["foliage_mass"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_sapwood_masses(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography.t_model_functions import calculate_sapwood_masses

        with outcome as excep:
            result = calculate_sapwood_masses(
                rho_s=rtmodel_flora.rho_s[pft_idx],
                ca_ratio=rtmodel_flora.ca_ratio[pft_idx],
                crown_area=rtmodel_data["crown_area"][data_idx],
                stem_height=rtmodel_data["stem_height"][data_idx],
                crown_fraction=rtmodel_data["crown_fraction"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["sapwood_mass"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_whole_crown_gpp(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of whole crown GPP."""

        from pyrealm.demography.t_model_functions import calculate_whole_crown_gpp

        with outcome as excep:
            result = calculate_whole_crown_gpp(
                lai=rtmodel_flora.lai[pft_idx],
                par_ext=rtmodel_flora.par_ext[pft_idx],
                crown_area=rtmodel_data["crown_area"][data_idx],
                potential_gpp=rtmodel_data["potential_gpp"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["whole_crown_gpp"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_sapwood_respiration(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of sapwood respiration."""

        from pyrealm.demography.t_model_functions import calculate_sapwood_respiration

        with outcome as excep:
            result = calculate_sapwood_respiration(
                resp_s=rtmodel_flora.resp_s[pft_idx],
                sapwood_mass=rtmodel_data["sapwood_mass"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["sapwood_respiration"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_foliar_respiration(
        self,
        request,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of foliar respiration.

        NOTE - this test is extremely circular, because the R implementation does not
        apply this in the same way. This is mostly just to validate array shape and
        broadcasting
        """

        from pyrealm.demography.t_model_functions import calculate_foliar_respiration

        with outcome as excep:
            result = calculate_foliar_respiration(
                resp_f=rtmodel_flora.resp_f[pft_idx],
                whole_crown_gpp=rtmodel_data["whole_crown_gpp"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(
                result,
                rtmodel_data["whole_crown_gpp"][data_idx]
                * rtmodel_flora.resp_f[pft_idx],
            )
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_fine_root_respiration(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of fine root respiration."""

        from pyrealm.demography.t_model_functions import calculate_fine_root_respiration

        with outcome as excep:
            result = calculate_fine_root_respiration(
                zeta=rtmodel_flora.zeta[pft_idx],
                sla=rtmodel_flora.sla[pft_idx],
                resp_r=rtmodel_flora.resp_r[pft_idx],
                foliage_mass=rtmodel_data["foliage_mass"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["fine_root_respiration"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_net_primary_productivity(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of net primary productivity."""

        from pyrealm.demography.t_model_functions import (
            calculate_net_primary_productivity,
        )

        with outcome as excep:
            result = calculate_net_primary_productivity(
                yld=rtmodel_flora.yld[pft_idx],
                whole_crown_gpp=rtmodel_data["whole_crown_gpp"][data_idx],
                foliar_respiration=0,  # Not included here in the R implementation
                fine_root_respiration=rtmodel_data["fine_root_respiration"][data_idx],
                sapwood_respiration=rtmodel_data["sapwood_respiration"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["npp"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_foliage_and_fine_root_turnover(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of foliage and fine root turnover."""

        from pyrealm.demography.t_model_functions import (
            calculate_foliage_and_fine_root_turnover,
        )

        with outcome as excep:
            result = calculate_foliage_and_fine_root_turnover(
                sla=rtmodel_flora.sla[pft_idx],
                zeta=rtmodel_flora.zeta[pft_idx],
                tau_f=rtmodel_flora.tau_f[pft_idx],
                tau_r=rtmodel_flora.tau_r[pft_idx],
                foliage_mass=rtmodel_data["foliage_mass"][data_idx],
            )

            assert result.shape == exp_shape
            assert np.allclose(result, rtmodel_data["turnover"][out_idx])
            return

        assert str(excep.value).startswith(excep_msg)

    def test_calculate_calculate_growth_increments(
        self,
        rtmodel_data,
        rtmodel_flora,
        data_idx,
        pft_idx,
        outcome,
        excep_msg,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of growth increments."""

        from pyrealm.demography.t_model_functions import (
            calculate_growth_increments,
        )

        with outcome as excep:
            delta_d, delta_mass_stm, delta_mass_frt = calculate_growth_increments(
                rho_s=rtmodel_flora.rho_s[pft_idx],
                a_hd=rtmodel_flora.a_hd[pft_idx],
                h_max=rtmodel_flora.h_max[pft_idx],
                lai=rtmodel_flora.lai[pft_idx],
                ca_ratio=rtmodel_flora.ca_ratio[pft_idx],
                sla=rtmodel_flora.sla[pft_idx],
                zeta=rtmodel_flora.zeta[pft_idx],
                npp=rtmodel_data["npp"][data_idx],
                turnover=rtmodel_data["turnover"][data_idx],
                dbh=rtmodel_data["dbh"][data_idx],
                stem_height=rtmodel_data["stem_height"][data_idx],
            )

            assert delta_d.shape == exp_shape
            assert np.allclose(delta_d, rtmodel_data["delta_dbh"][out_idx])

            assert delta_mass_stm.shape == exp_shape
            assert np.allclose(delta_mass_stm, rtmodel_data["delta_stem_mass"][out_idx])

            assert delta_mass_frt.shape == exp_shape
            assert np.allclose(
                delta_mass_frt, rtmodel_data["delta_foliage_mass"][out_idx]
            )
            return

        assert str(excep.value).startswith(excep_msg)


def test_calculate_dbh_from_height_edge_cases():
    """Test inverted calculation of dbh from height handles edges cases.

    * If H > h_max, dbh is not calculable and should be np.nan
    * If H = h_max, dbh is infinite.
    """

    from pyrealm.demography.t_model_functions import calculate_dbh_from_height

    pft_h_max_values = np.array([20, 30])
    pft_a_hd_values = np.array([116.0, 116.0])
    stem_heights = np.array([[0], [10], [20], [30], [40]])

    dbh = calculate_dbh_from_height(
        h_max=pft_h_max_values,
        a_hd=pft_a_hd_values,
        stem_height=stem_heights,
    )

    # first row should be all zeros (zero height gives zero diameter)
    assert np.all(dbh[0, :] == 0)

    # Infinite entries
    assert np.all(np.isinf(dbh) == np.array([[0, 0], [0, 0], [1, 0], [0, 1], [0, 0]]))

    # Undefined entries
    assert np.all(np.isnan(dbh) == np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 1]]))


def test_StemAllometry(rtmodel_flora, rtmodel_data):
    """Test the StemAllometry class."""

    from pyrealm.demography.t_model_functions import StemAllometry

    stem_allometry = StemAllometry(
        stem_traits=rtmodel_flora, at_dbh=rtmodel_data["dbh"][:, [0]]
    )

    # Check the variables provided by the rtmodel implementation
    vars_to_check = (
        v
        for v in stem_allometry.allometry_attrs
        if v not in ["crown_r0", "crown_z_max"]
    )
    for var in vars_to_check:
        assert np.allclose(getattr(stem_allometry, var), rtmodel_data[var])


def test_StemAllocation(rtmodel_flora, rtmodel_data):
    """Test the StemAllometry class."""

    from pyrealm.demography.t_model_functions import StemAllocation, StemAllometry

    stem_allometry = StemAllometry(
        stem_traits=rtmodel_flora, at_dbh=rtmodel_data["dbh"][:, [0]]
    )

    stem_allocation = StemAllocation(
        stem_traits=rtmodel_flora,
        stem_allometry=stem_allometry,
        at_potential_gpp=rtmodel_data["potential_gpp"],
    )

    # Check the variables provided by the rtmodel implementation
    vars_to_check = (
        v for v in stem_allocation.allocation_attrs if v not in ["foliar_respiration"]
    )
    for var in vars_to_check:
        assert np.allclose(getattr(stem_allocation, var), rtmodel_data[var])
