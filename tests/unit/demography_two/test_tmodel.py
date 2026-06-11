"""test the functions in tmodel.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Test functions that are not part of the original rtmodel.


@pytest.mark.parametrize(
    argnames="crown_areas, expected_r0",
    argvalues=(
        (np.array([20, 30]), np.array([0.86887756, 1.29007041])),
        (np.array([30, 40]), np.array([1.06415334, 1.489645])),
    ),
)
def test_calculate_crown_r_0_values(crown_areas, expected_r0):
    """Test happy path for calculating r_0."""

    from pyrealm.demography_two.tmodel import calculate_crown_r0

    q_m = np.array([2.9038988210485766, 2.3953681843215673])
    actual_r0_values = calculate_crown_r0(q_m=q_m, crown_area=crown_areas)

    assert_allclose(actual_r0_values, expected_r0)


def test_calculate_reproductive_tissue_mass():
    """Tests calculation of reproductive tissue mass."""

    from pyrealm.demography_two.tmodel import calculate_reproductive_tissue_mass

    result = calculate_reproductive_tissue_mass(
        p_foliage_for_reproductive_tissue=np.array([10]),
        foliage_mass=np.array([0.25]),
    )

    assert result == np.array([2.5])


def test_calculate_gpp_topslice():
    """Tests calculation of gpp_topslice."""

    from pyrealm.demography_two.tmodel import calculate_gpp_topslice

    result = calculate_gpp_topslice(
        gpp_topslice=np.array([10]),
        whole_crown_gpp=np.array([0.25]),
    )

    assert result == np.array([2.5])


def test_calculate_reproductive_tissue_respiration():
    """Tests calculation of reproductive tissue respiration."""

    from pyrealm.demography_two.tmodel import calculate_reproductive_tissue_respiration

    result = calculate_reproductive_tissue_respiration(
        resp_rt=np.array([10]),
        reproductive_tissue_mass=np.array([0.25]),
    )

    assert result == np.array([2.5])


def test_calculate_reproductive_tissue_turnover():
    """Tests calculation of reproductive tissue turnover."""

    from pyrealm.demography_two.tmodel import calculate_reproductive_tissue_turnover

    result = calculate_reproductive_tissue_turnover(
        reproductive_tissue_mass=np.array([10]),
        tau_rt=np.array([4]),
    )

    assert result == np.array([2.5])


@pytest.mark.parametrize(
    argnames="dbh_idx, data_idx, out_idx, exp_shape",
    argvalues=[
        pytest.param(0, (0, ...), 0, (3,), id="scalar"),
        pytest.param((0, ...), (0, ...), 0, (3,), id="row_0"),
        pytest.param((1, ...), (1, ...), 1, (3,), id="row_1"),
        pytest.param((2, ...), (2, ...), 2, (3,), id="row_2"),
        pytest.param((3, ...), (3, ...), 3, (3,), id="row_3"),
        pytest.param((4, ...), (4, ...), 4, (3,), id="row_4"),
        pytest.param((5, ...), (5, ...), 5, (3,), id="row_5"),
        pytest.param((..., [0]), tuple(), tuple(), (6, 3), id="column"),
        pytest.param(tuple(), tuple(), tuple(), (6, 3), id="full"),
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
    inputs. This doesn't exhaustively test all failure modes - there is a more detailed
    test of _validate_demography_array_arguments in tests/unit/demography/test_core.py
    """

    def test_calculate_heights(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of heights of tree from diameter."""
        from pyrealm.demography_two.tmodel import calculate_heights

        result = calculate_heights(
            h_max=np.array(rtmodel_flora.h_max),
            a_hd=np.array(rtmodel_flora.a_hd),
            dbh=rtmodel_data["dbh"][dbh_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["stem_height"][out_idx])

    def test_calculate_dbh_from_height(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests inverted calculation of dbh from height."""

        from pyrealm.demography_two.tmodel import calculate_dbh_from_height

        result = calculate_dbh_from_height(
            h_max=np.array(rtmodel_flora.h_max),
            a_hd=np.array(rtmodel_flora.a_hd),
            stem_height=rtmodel_data["stem_height"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["dbh"][out_idx])

    def test_calculate_crown_areas(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of crown areas of trees."""

        from pyrealm.demography_two.tmodel import calculate_crown_areas

        result = calculate_crown_areas(
            ca_ratio=np.array(rtmodel_flora.ca_ratio),
            a_hd=np.array(rtmodel_flora.a_hd),
            dbh=rtmodel_data["dbh"][dbh_idx],
            stem_height=rtmodel_data["stem_height"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["crown_area"][out_idx])

    def test_calculate_crown_fractions(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of crown fraction of trees."""

        from pyrealm.demography_two.tmodel import calculate_crown_fractions

        result = calculate_crown_fractions(
            a_hd=np.array(rtmodel_flora.a_hd),
            dbh=rtmodel_data["dbh"][dbh_idx],
            stem_height=rtmodel_data["stem_height"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["crown_fraction"][out_idx])

    def test_calculate_stem_masses(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography_two.tmodel import calculate_stem_masses

        result = calculate_stem_masses(
            rho_s=np.array(rtmodel_flora.rho_s),
            dbh=rtmodel_data["dbh"][data_idx],
            stem_height=rtmodel_data["stem_height"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["stem_mass"][out_idx])

    def test_calculate_foliage_masses(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography_two.tmodel import calculate_foliage_masses

        result = calculate_foliage_masses(
            lai=np.array(rtmodel_flora.lai),
            sla=np.array(rtmodel_flora.sla),
            crown_area=rtmodel_data["crown_area"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["foliage_mass"][out_idx])

    def test_calculate_sapwood_masses(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of stem masses of trees."""

        from pyrealm.demography_two.tmodel import calculate_sapwood_masses

        result = calculate_sapwood_masses(
            rho_s=np.array(rtmodel_flora.rho_s),
            ca_ratio=np.array(rtmodel_flora.ca_ratio),
            crown_area=rtmodel_data["crown_area"][data_idx],
            stem_height=rtmodel_data["stem_height"][data_idx],
            crown_fraction=rtmodel_data["crown_fraction"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["sapwood_mass"][out_idx])

    def test_calculate_whole_crown_gpp(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of whole crown GPP."""

        from pyrealm.demography_two.tmodel import calculate_whole_crown_gpp

        result = calculate_whole_crown_gpp(
            lai=np.array(rtmodel_flora.lai),
            par_ext=np.array(rtmodel_flora.par_ext),
            crown_area=rtmodel_data["crown_area"][data_idx],
            potential_gpp=rtmodel_data["potential_gpp"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["whole_crown_gpp"][out_idx])

    def test_calculate_sapwood_respiration(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of sapwood respiration."""

        from pyrealm.demography_two.tmodel import calculate_sapwood_respiration

        result = calculate_sapwood_respiration(
            resp_s=np.array(rtmodel_flora.resp_s),
            sapwood_mass=rtmodel_data["sapwood_mass"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["sapwood_respiration"][out_idx])

    def test_calculate_foliage_respiration(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of foliage respiration.

        NOTE - this test is extremely circular, because the R implementation does not
        apply this in the same way. This is mostly just to validate array shape and
        broadcasting
        """

        from pyrealm.demography_two.tmodel import calculate_foliage_respiration

        result = calculate_foliage_respiration(
            resp_f=np.array(rtmodel_flora.resp_f),
            whole_crown_gpp=rtmodel_data["whole_crown_gpp"][data_idx],
        )

        assert result.shape == exp_shape
        assert_allclose(
            result,
            rtmodel_data["whole_crown_gpp"][out_idx] * np.array(rtmodel_flora.resp_f),
        )

    def test_calculate_fine_root_respiration(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of fine root respiration.

        Because this uses an intermediate calculation, the failure modes are triggered
        by that intermediate calculation rather than calculate_fine_root_respiration.
        """

        from pyrealm.demography_two.tmodel import (
            calculate_fine_root_masses,
            calculate_fine_root_respiration,
        )

        # Original implementation does not store fine root mass so calculate
        # required intermediate variable
        fine_root_mass = calculate_fine_root_masses(
            zeta=np.array(rtmodel_flora.zeta),
            lai=np.array(rtmodel_flora.lai),
            crown_area=rtmodel_data["crown_area"][data_idx],
        )

        result = calculate_fine_root_respiration(
            resp_r=np.array(rtmodel_flora.resp_r),
            fine_root_mass=fine_root_mass,
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["fine_root_respiration"][out_idx])

    def test_calculate_net_primary_productivity(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of net primary productivity."""

        from pyrealm.demography_two.tmodel import (
            calculate_net_primary_productivity,
        )

        result = calculate_net_primary_productivity(
            yld=np.array(rtmodel_flora.yld),
            whole_crown_gpp=rtmodel_data["whole_crown_gpp"][data_idx],
            foliage_respiration=np.array([0]),
            fine_root_respiration=rtmodel_data["fine_root_respiration"][data_idx],
            sapwood_respiration=rtmodel_data["sapwood_respiration"][data_idx],
            reproductive_tissue_respiration=np.zeros(
                np.shape(rtmodel_data["sapwood_respiration"][data_idx])
            ),
        )

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["npp"][out_idx])

    def test_calculate_turnover(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of foliage and fine root turnover.

        Because this uses an intermediate calculation, the failure modes are triggered
        by that intermediate calculation rather than calculate_fine_root_turnover.
        """

        from pyrealm.demography_two.tmodel import (
            calculate_fine_root_masses,
            calculate_fine_root_turnover,
            calculate_foliage_turnover,
        )

        # Original implementation does not store fine root mass so calculate
        # required intermediate variable

        fine_root_mass = calculate_fine_root_masses(
            zeta=np.array(rtmodel_flora.zeta),
            lai=np.array(rtmodel_flora.lai),
            crown_area=rtmodel_data["crown_area"][data_idx],
        )

        result1 = calculate_fine_root_turnover(
            tau_r=np.array(rtmodel_flora.tau_r), fine_root_mass=fine_root_mass
        )

        result2 = calculate_foliage_turnover(
            tau_f=np.array(rtmodel_flora.tau_f),
            foliage_mass=rtmodel_data["foliage_mass"][data_idx],
        )

        result = result1 + result2

        assert result.shape == exp_shape
        assert_allclose(result, rtmodel_data["turnover"][out_idx])

    def test_calculate_growth_increments(
        self,
        rtmodel_data,
        rtmodel_flora,
        dbh_idx,
        data_idx,
        out_idx,
        exp_shape,
    ):
        """Tests calculation of growth increments."""

        from pyrealm.demography_two.tmodel import (
            calculate_growth_increments,
        )

        delta_d, delta_mass_stm, delta_mass_f, delta_mass_rt = (
            calculate_growth_increments(
                rho_s=np.array(rtmodel_flora.rho_s),
                a_hd=np.array(rtmodel_flora.a_hd),
                h_max=np.array(rtmodel_flora.h_max),
                lai=np.array(rtmodel_flora.lai),
                ca_ratio=np.array(rtmodel_flora.ca_ratio),
                sla=np.array(rtmodel_flora.sla),
                zeta=np.array(rtmodel_flora.zeta),
                npp=rtmodel_data["npp"][data_idx],
                turnover=rtmodel_data["turnover"][data_idx],
                reproductive_tissue_turnover=np.zeros(
                    np.shape(rtmodel_data["turnover"][data_idx])
                ),
                p_foliage_for_reproductive_tissue=np.zeros(
                    np.shape(rtmodel_data["npp"][data_idx])
                ),
                dbh=rtmodel_data["dbh"][data_idx],
                stem_height=rtmodel_data["stem_height"][data_idx],
            )
        )

        assert delta_d.shape == exp_shape
        assert_allclose(delta_d, rtmodel_data["delta_dbh"][out_idx])

        assert delta_mass_stm.shape == exp_shape
        assert_allclose(delta_mass_stm, rtmodel_data["delta_stem_mass"][out_idx])

        assert delta_mass_f.shape == exp_shape
        assert delta_mass_rt.shape == exp_shape
        assert_allclose(
            delta_mass_f + delta_mass_rt,
            rtmodel_data["delta_foliage_mass"][out_idx],
        )


def test_calculate_dbh_from_height_edge_cases():
    """Test inverted calculation of dbh from height handles edges cases.

    * If H > h_max, dbh is not calculable and should be np.nan
    * If H = h_max, dbh is infinite.
    """

    from pyrealm.demography_two.tmodel import calculate_dbh_from_height

    pft_h_max_values = np.array([20, 30])
    pft_a_hd_values = np.array([116.0, 116.0])
    stem_heights = np.array([[10], [20], [30], [40]])

    dbh = calculate_dbh_from_height(
        h_max=pft_h_max_values,
        a_hd=pft_a_hd_values,
        stem_height=stem_heights,
    )

    # Infinite entries
    assert np.all(np.isinf(dbh) == np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))

    # Undefined entries
    assert np.all(np.isnan(dbh) == np.array([[0, 0], [0, 0], [1, 0], [1, 1]]))


@pytest.mark.parametrize(
    argnames="dbh_idx, at_dbh, data_idx, out_idx, exp_shape",
    argvalues=[
        pytest.param((0, ...), False, (0, ...), 0, (3,), id="row_0"),
        pytest.param((1, ...), False, (1, ...), 1, (3,), id="row_1"),
        pytest.param((2, ...), False, (2, ...), 2, (3,), id="row_2"),
        pytest.param((3, ...), False, (3, ...), 3, (3,), id="row_3"),
        pytest.param((4, ...), False, (4, ...), 4, (3,), id="row_4"),
        pytest.param((5, ...), False, (5, ...), 5, (3,), id="row_5"),
        pytest.param((0, ...), True, tuple(), tuple(), (6, 3), id="use_at_dbh"),
    ],
)
def test_StemAllometry(
    rtmodel_flora, rtmodel_data, dbh_idx, at_dbh, data_idx, out_idx, exp_shape
):
    """Test the StemAllometry class and inherited methods."""

    from pyrealm.demography_two.cohorts import CohortData, Cohorts
    from pyrealm.demography_two.tmodel import StemAllometry

    cohort_data = CohortData(
        pft_name=rtmodel_flora.name,
        n_individuals=[1, 1, 1],
        dbh_value=rtmodel_data["dbh"][dbh_idx],
    )
    cohorts = Cohorts(cohort_data=cohort_data, flora=rtmodel_flora)

    at_dbh_value = rtmodel_data["dbh"][:, 0] if at_dbh else None
    stem_allometry = StemAllometry(cohorts=cohorts, at_dbh=at_dbh_value)

    # Check the values of the variables calculated against the expectations from the
    # rtmodel implementation
    vars_to_check = (
        v
        for v in stem_allometry.array_attrs
        if v
        not in [
            "dbh",  # TODO NEED TO ADD THIS BACK IN!
            "crown_r0",
            "crown_z_max",
            "reproductive_tissue_mass",
            "fine_root_mass",
        ]
    )
    for var in vars_to_check:
        assert_allclose(getattr(stem_allometry, var), rtmodel_data[var][out_idx])

    # # Test the inherited to_pandas method
    # df = stem_allometry.to_pandas()

    # assert df.shape == (
    #     stem_allometry._n_stems * stem_allometry._n_pred,
    #     len(stem_allometry.array_attrs),
    # )

    # assert set(stem_allometry.array_attrs) == set(df.columns)


# def test_StemAllometry_CohortMethods(rtmodel_flora, rtmodel_data):
#     """Test the StemAllometry inherited cohort methods."""

#     from pyrealm.demography_two.tmodel import StemAllometry

#     stem_allometry = StemAllometry(
#         stem_traits=rtmodel_flora, at_dbh=rtmodel_data["dbh"][:, [0]]
#     )
#     check_data = stem_allometry.crown_fraction.copy()

#     # Check the count attribute
#     assert stem_allometry._n_stems == rtmodel_flora.n_pfts

#     # Check failure mode
#     with pytest.raises(ValueError) as excep:
#         stem_allometry.add_cohort_data(new_data=dict(a=1))

#     assert (
#         str(excep.value)
#         == "Cannot add cohort data from an dict instance to StemAllometry"
#     )

#     # Check success of adding and dropping data
#     n_entries = len(rtmodel_data["dbh"])
#     # Add a copy of itself as new cohort data and check the shape
#     stem_allometry.add_cohort_data(new_data=stem_allometry)
#     assert stem_allometry.crown_fraction.shape == (
#           n_entries, 2 * rtmodel_flora.n_pfts
#     )
#     assert stem_allometry.crown_fraction.sum() == 2 * check_data.sum()
#     assert stem_allometry._n_stems == 2 * rtmodel_flora.n_pfts

#     # Remove the rows from the first copy and what's left should be aligned with the
#     # original data
#     stem_allometry.drop_cohort_data(drop_indices=np.arange(rtmodel_flora.n_pfts))
#     assert_allclose(stem_allometry.crown_fraction, check_data)
#     assert stem_allometry._n_stems == rtmodel_flora.n_pfts


# def test_StemAllocation(rtmodel_flora, rtmodel_data):
#     """Test the StemAllometry class."""

#     from pyrealm.demography_two.tmodel import (
#         StemAllocation,
#         StemAllometry,
#         calculate_whole_crown_gpp,
#     )

#     stem_allometry = StemAllometry(
#         stem_traits=rtmodel_flora, at_dbh=rtmodel_data["dbh"][:, [0]]
#     )

#     # Calculate the Li et al Equation 12 GPP from the P0 values
#     whole_crown_gpp = calculate_whole_crown_gpp(
#         potential_gpp=rtmodel_data["potential_gpp"],
#         crown_area=stem_allometry.crown_area,
#         par_ext=rtmodel_flora.par_ext,
#         lai=rtmodel_flora.lai,
#     )

#     stem_allocation = StemAllocation(
#         stem_traits=rtmodel_flora,
#         stem_allometry=stem_allometry,
#         whole_crown_gpp=whole_crown_gpp,
#     )

#     # Check the values of the variables calculated against the expectations from the
#     # rtmodel implementation
#     vars_to_check = (
#         v
#         for v in stem_allocation.array_attrs
#         if v
#         not in [
#             "foliar_respiration",
#             "foliage_turnover",
#             "fine_root_turnover",
#             "reproductive_tissue_respiration",
#             "reproductive_tissue_turnover",
#             "delta_foliage_mass",
#             "delta_fine_root_mass",
#         ]
#     )
#     for var in vars_to_check:
#         assert_allclose(getattr(stem_allocation, var), rtmodel_data[var])

#     # Separately check the partitioning into delta foliage and fine root
#     assert_allclose(
#         stem_allocation.delta_foliage_mass + stem_allocation.delta_fine_root_mass,
#         rtmodel_data["delta_foliage_mass"],
#     )

#     # Test the inherited to_pandas method
#     df = stem_allocation.to_pandas()

#     assert df.shape == (
#         stem_allocation._n_stems * stem_allocation._n_pred,
#         len(stem_allocation.array_attrs),
#     )

#     assert set(stem_allocation.array_attrs) == set(df.columns)


# @pytest.mark.parametrize(
#     argnames="whole_crown_gpp, outcome, excep_msg",
#     argvalues=[
#         pytest.param(np.array(1), does_not_raise(), None, id="pass_0D"),
#         pytest.param(np.ones(1), does_not_raise(), None, id="pass_1D_scalar"),
#         pytest.param(np.ones(3), does_not_raise(), None, id="pass_1D_row"),
#         pytest.param(np.ones((1, 3)), does_not_raise(), None, id="pass_2D_row"),
#         pytest.param(np.ones((4, 1)), does_not_raise(), None, id="pass_2D_col"),
#         pytest.param(np.ones((4, 3)), does_not_raise(), None, id="pass_2D_full"),
#         pytest.param(
#             np.ones(4),
#             pytest.raises(ValueError),
#             "The broadcast shapes of the trait and size arguments (4, 3) are not "
#             "congruent with the shape of the at_size arguments (4,)",
#             id="fail_1D_row_wrong",
#         ),
#         pytest.param(
#             np.ones((5, 4)),
#             pytest.raises(ValueError),
#             "The broadcast shapes of the trait and size arguments (4, 3) are not "
#             "congruent with the shape of the at_size arguments (5, 4)",
#             id="fail_2D_full_wrong",
#         ),
#     ],
# )
# def test_StemAllocation_validation(
#       rtmodel_flora, whole_crown_gpp, outcome, excep_msg
# ):
#     """Test the StemAllocation validation process.

#     The stem allometry inputs are kept constant - the validation of inputs to that
#     function is checked in the tests above.

#     Also checks that validation only occurs once. Note that this requires the use of
#     the
#     wraps argument to ensure that the validation function actually _runs_ while being
#     spied on, rather than just being replaced by the patch.
#     """

#     from pyrealm.demography_two.core import _validate_demography_array_arguments
#     from pyrealm.demography_two.tmodel import StemAllocation, StemAllometry

#     # Calculate a constant allometry
#     allom = StemAllometry(stem_traits=rtmodel_flora, at_dbh=np.ones((4, 3)))

#     with (
#         outcome as excep,
#         patch(
#             "pyrealm.demography_two.tmodel._validate_demography_array_arguments",
#             wraps=_validate_demography_array_arguments,
#         ) as val_func_patch,
#     ):
#         # Check the behaviour of the validation
#         _ = StemAllocation(
#             stem_traits=rtmodel_flora,
#             stem_allometry=allom,
#             whole_crown_gpp=whole_crown_gpp,
#         )
#         assert val_func_patch.call_count == 1
#         return

#     assert str(excep.value).startswith(excep_msg)


# @pytest.mark.parametrize(
#     argnames="dbh, outcome, msg",
#     argvalues=(
#         pytest.param(np.array([3, 3, 3]), does_not_raise(), None, id="positive"),
#         pytest.param(
#             np.array([0, 0, 3]),
#             pytest.raises(ValueError),
#             "Allometry values in StemAllometry not strictly positive: at_dbh",
#             id="zero",
#         ),
#         pytest.param(
#             np.array([-3, -2, 3]),
#             pytest.raises(ValueError),
#             "Allometry values in StemAllometry not strictly positive: at_dbh",
#             id="negative",
#         ),
#     ),
# )
# def test_stem_allocation_strictly_positive_sizes(rtmodel_flora, dbh, outcome, msg):
#     """Test that StemAllometry handles zero and negative DBH."""

#     from pyrealm.demography_two.tmodel import StemAllometry

#     with outcome as excep:
#         # Calculate allometry for zero DBH stems of each PFT
#         _ = StemAllometry(stem_traits=rtmodel_flora, at_dbh=dbh)
#         return

#     assert str(excep.value) == msg
