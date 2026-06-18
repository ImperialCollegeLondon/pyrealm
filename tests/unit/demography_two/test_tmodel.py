"""test the functions in tmodel.py."""

from contextlib import nullcontext as does_not_raise

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
    of the T model functions. The combination of indices slice up the inputs to test
    a wide range of input shape combinations.

    The PFT trait data is expected to always come in as a row array, which is what is
    provided by the rtmodel_flora object. The other inputs then depend on the shape of
    the initial DBH values:

    * If there is a single value for DBH (scalar) or a (matching!) row array, then all
      of the data will retain the shape of the initial trait data:
            (1,) * (3, ) and (3, ) (3,) * (3, ) -> (3, )
    * If the DBH is a column array, then the data are broadcast to the outer product:
            (6, 1) * (3, ) -> (6, 3)
    * If the DBH is a 2D array that broadcasts along the second dimension, the data will
      match the DBH dims - this is not expected to be a common use mode as the
      StemAllometry and StemAllocation classes deliberately do not support it.
            (6, 3) * (3, ) -> (6, 3)

    For other inputs that are generated from initial DBH, the input indices for
    validation need to to match those expected output shapese from DBH. This is why
    there are both dbh_idx and data_idx parameters. The out_idx is used to select the
    expected slice from the test values.
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
    argnames="dbh_idx, at_dbh, data_idx, out_idx",
    argvalues=[
        pytest.param((0, ...), False, (0, ...), 0, id="row_0"),
        pytest.param((1, ...), False, (1, ...), 1, id="row_1"),
        pytest.param((2, ...), False, (2, ...), 2, id="row_2"),
        pytest.param((3, ...), False, (3, ...), 3, id="row_3"),
        pytest.param((4, ...), False, (4, ...), 4, id="row_4"),
        pytest.param((5, ...), False, (5, ...), 5, id="row_5"),
        # Note that the dbh_idx below is arbitrary as it is overridden using at_dbh.
        pytest.param((0, ...), True, tuple(), tuple(), id="use_at_dbh"),
    ],
)
def test_StemAllometry_and_StemAllocation(
    rtmodel_flora, rtmodel_data, dbh_idx, at_dbh, data_idx, out_idx
):
    """Test the StemAllometry, StemAllocation classes and inherited methods.

    This test checks that the StemAllometry and StemAllocation classes generate the same
    predictions as the original R model.

    It also also checks the dimensionality of StemAllometry _only_ by producing
    results in "standard mode" - just using the cohort DBH values - and "profile mode"
    when using `at_dbh` to get 2D allometries.

    It doesn't test the dimensionality of StemAllocation (see below) beyond checking
    that 2D allometries generate the correct prediction.
    """

    from pyrealm.demography_two.cohorts import CohortData, Cohorts
    from pyrealm.demography_two.tmodel import (
        StemAllocation,
        StemAllometry,
        calculate_whole_crown_gpp,
    )

    ## Generate cohort data
    cohort_data = CohortData(
        pft_name=rtmodel_flora.name,
        n_individuals=[1, 1, 1],
        dbh_value=rtmodel_data["dbh"][dbh_idx],
    )
    cohorts = Cohorts(cohort_data=cohort_data, flora=rtmodel_flora)

    ## Check that StemAllometry produces the correct values

    at_dbh_value = rtmodel_data["dbh"][:, 0] if at_dbh else None
    stem_allometry = StemAllometry(cohorts=cohorts, at_dbh=at_dbh_value)

    # Check the values of the variables calculated against the expectations from the
    # rtmodel implementation - also checks shape of outputs
    vars_to_check = (
        v
        for v in stem_allometry._array_attrs
        if v
        not in [
            "cohort_ids",
            "crown_r0",
            "crown_z_max",
            "reproductive_tissue_mass",
            "fine_root_mass",
        ]
    )
    for var in vars_to_check:
        assert_allclose(getattr(stem_allometry, var), rtmodel_data[var][out_idx])

    # Test the ToDataFrameMixin.to_dataframe() method
    df = stem_allometry.to_dataframe()

    assert df.shape == (
        np.prod(stem_allometry.dbh.shape),
        len(stem_allometry._array_attrs),
    )

    assert set(stem_allometry._array_attrs) == set(df.columns)

    ## Check that StemAllocation produces the correct values

    # Calculate the Li et al Equation 12 GPP from the P0 values
    whole_crown_gpp = calculate_whole_crown_gpp(
        potential_gpp=np.array(rtmodel_data["potential_gpp"][data_idx]),
        crown_area=stem_allometry.crown_area,
        par_ext=np.array(rtmodel_flora.par_ext),
        lai=np.array(rtmodel_flora.lai),
    )

    stem_allocation = StemAllocation(
        cohorts=cohorts,
        allometry=stem_allometry,
        whole_crown_gpp=whole_crown_gpp,
    )

    # Check the values of the variables calculated against the expectations from the
    # rtmodel implementation
    vars_to_check = (
        v
        for v in stem_allocation._array_attrs
        if v
        not in [
            "cohort_ids",
            "foliage_respiration",
            "foliage_turnover",
            "fine_root_turnover",
            "reproductive_tissue_respiration",
            "reproductive_tissue_turnover",
            "delta_foliage_mass",
            "delta_fine_root_mass",
        ]
    )
    for var in vars_to_check:
        assert_allclose(getattr(stem_allocation, var), rtmodel_data[var][out_idx])

    # Separately check the partitioning into delta foliage and fine root
    assert_allclose(
        stem_allocation.delta_foliage_mass + stem_allocation.delta_fine_root_mass,
        rtmodel_data["delta_foliage_mass"][out_idx],
    )

    # Test the ToDataFrameMixin.to_dataframe() method
    df = stem_allocation.to_dataframe()

    assert df.shape == (
        np.prod(stem_allocation.sapwood_respiration.shape),
        len(stem_allocation._array_attrs),
    )

    assert set(stem_allocation._array_attrs) == set(df.columns)


@pytest.mark.parametrize(
    argnames="at_dbh, gpp, profile, exp_shape, df_rows",
    argvalues=(
        pytest.param(None, np.ones(1), False, (3,), 3, id="standard_mode_scalar"),
        pytest.param(None, np.ones(3), False, (3,), 3, id="standard_mode"),
        pytest.param(np.ones(4), np.ones(1), False, (4, 3), 12, id="at_dbh_gpp_scalar"),
        pytest.param(
            np.ones(4), np.ones(3), False, (4, 3), 12, id="at_dbh_gpp_per_cohort"
        ),
        pytest.param(
            np.ones(4), np.ones((4, 3)), False, (4, 3), 12, id="atdbh_gpp_per_element"
        ),
        pytest.param(None, np.ones(5), True, (5, 3), 15, id="1d_allom_profile"),
        pytest.param(
            np.ones(4), np.ones(5), True, (5, 4, 3), 60, id="2d_allom_profile"
        ),
    ),
)
def test_StemAllocation_GPP_inputs(
    rtmodel_flora, at_dbh, gpp, profile, exp_shape, df_rows
):
    """Test the dimensionality of StemAllocation outputs.

    The parameterisation provides tests of output shapes under different operating
    modes.
    """
    from pyrealm.demography_two.cohorts import CohortData, Cohorts
    from pyrealm.demography_two.tmodel import (
        StemAllocation,
        StemAllometry,
    )

    ## Generate cohort data and calculate the allometry and allocation
    cohort_data = CohortData(
        pft_name=rtmodel_flora.name,
        n_individuals=[1, 1, 1],
        dbh_value=[0.5, 0.5, 0.5],
    )
    cohorts = Cohorts(cohort_data=cohort_data, flora=rtmodel_flora)
    allom = StemAllometry(cohorts=cohorts, at_dbh=at_dbh)
    alloc = StemAllocation(
        cohorts=cohorts, allometry=allom, whole_crown_gpp=gpp, profile=profile
    )

    # Check the attribute shape
    assert alloc.whole_crown_gpp.shape == exp_shape
    assert alloc.cohort_ids.shape == exp_shape
    assert alloc.foliage_respiration.shape == exp_shape

    # Check dataframe conversion and repr works
    with does_not_raise():
        df = alloc.to_dataframe()
        assert df.shape == (df_rows, len(alloc._array_attrs))
        repr(alloc)
