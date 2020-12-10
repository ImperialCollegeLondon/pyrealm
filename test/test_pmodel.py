import pytest
import numpy as np
import yaml
from contextlib import contextmanager
from pypmodel import pmodel

# ------------------------------------------
# Null context manager to include exception testing in test paramaterisation
# ------------------------------------------


@contextmanager
def does_not_raise():
    yield

# ------------------------------------------
# Fixtures: inputs and expected values
# ------------------------------------------


@pytest.fixture(scope='module')
def inputs():
    """Fixture to load test inputs from file.
    """

    with open('test/test_inputs.yaml') as infile:
        inputs = yaml.load(infile, Loader=yaml.SafeLoader)

    inputs = {k: np.array(v) if isinstance(v, list) else v
              for k, v in inputs.items()}

    return inputs


@pytest.fixture(scope='module')
def expected():
    """Fixture to load expectations from rpmodel from file
    """

    with open('test/test_outputs_rpmodel.yaml') as infile:
        expected = yaml.load(infile, Loader=yaml.SafeLoader)

    expected = {k: np.array(v) if isinstance(v, list) else v
                for k, v in expected.items()}

    return expected

# ------------------------------------------
# Test structure
# The basic structure of these tests  is to use a pytest.mark.parameterise
# to pass in a dictionary of the variables to be passed to the function and then
# a dictionary identifying the expected values that have been read in from file.
#
# A separate R file is used to read in the same inputs to R and run them through
# rpmodel to generate the output file.
#
# TODO - should be able to group shared inputs (e.g. tc + patm) to recycle
#        parameterisation.
# ------------------------------------------

# ------------------------------------------
# Testing calc_density_h20 - temp + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc', patm='patm'),  # scalars
           cmng=does_not_raise(),
           out='dens_h20_sc')),
     (dict(args=dict(tc='tc', patm='patm_mat'),  # scalar + array
           cmng=does_not_raise(),
           out='dens_h20_sc_ar')),
     (dict(args=dict(tc='tc_mat', patm='patm_mat'),  # arrays
           cmng=does_not_raise(),
           out='dens_h20_ar')),
     (dict(args=dict(tc='tc_mat', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_co2_to_ca(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_density_h2o(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_ftemp_inst_rd - temp only but in Kelvin!
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tk='tk', dha='KattgeKnorr_ha'),  # scalar
           cmng=does_not_raise(),
           out='ftemp_arrh_sc')),
     (dict(args=dict(tk='tk_mat', dha='KattgeKnorr_ha'),  # array
           cmng=does_not_raise(),
           out='ftemp_arrh_ar'))]
)
def test_calc_ftemp_arrh(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_ftemp_arrh(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_ftemp_inst_vcmax - temp only
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc'),  # scalar
           cmng=does_not_raise(),
           out='ftemp_inst_vcmax_sc')),
     (dict(args=dict(tc='tc_mat'),  # array
           cmng=does_not_raise(),
           out='ftemp_inst_vcmax_ar'))]
)
def test_calc_ftemp_inst_vcmax(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_ftemp_inst_vcmax(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_ftemp_inst_vcmax - temp only
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc'), c4=False,  # scalar, C3
           cmng=does_not_raise(),
           out='ftemp_kphio_c3_sc')),
     (dict(args=dict(tc='tc_mat'), c4=False,  # array, C3
           cmng=does_not_raise(),
           out='ftemp_kphio_c3_ar')),
     (dict(args=dict(tc='tc'), c4=True,  # scalar, C4
           cmng=does_not_raise(),
           out='ftemp_kphio_c4_sc')),
     (dict(args=dict(tc='tc_mat'), c4=True,  # array, C4
           cmng=does_not_raise(),
           out='ftemp_kphio_c4_ar'))]
)
def test_calc_ftemp_kphio(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_ftemp_kphio(**kwargs, c4=ctrl['c4'])
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_gammastar - temp + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc', patm='patm'),  # scalars
           cmng=does_not_raise(),
           out='gammastar_sc')),
     (dict(args=dict(tc='tc_mat', patm='patm'),  # scalar + array
           cmng=does_not_raise(),
           out='gammastar_sc_ar')),
     (dict(args=dict(tc='tc_mat', patm='patm_mat'),  # arrays
           cmng=does_not_raise(),
           out='gammastar_ar')),
     (dict(args=dict(tc='tc_mat', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_gammastar(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_gammastar(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_kmm - temp + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc', patm='patm'),  # scalars
           cmng=does_not_raise(),
           out='kmm_sc')),
     (dict(args=dict(tc='tc_mat', patm='patm'),  # scalar + array
           cmng=does_not_raise(),
           out='kmm_sc_ar')),
     (dict(args=dict(tc='tc_mat', patm='patm_mat'),  # arrays
           cmng=does_not_raise(),
           out='kmm_ar')),
     (dict(args=dict(tc='tc_mat', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_kmm(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_kmm(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_soilmstress - soilm + meanalpha
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(soilm='soilm', meanalpha='meanalpha'),  # scalars
           cmng=does_not_raise(),
           out='soilmstress_sc')),
     (dict(args=dict(soilm='soilm_mat', meanalpha='meanalpha'),  # scalar + array
           cmng=does_not_raise(),
           out='soilmstress_sc_ar')),
     (dict(args=dict(soilm='soilm_mat', meanalpha='meanalpha_mat'),  # arrays
           cmng=does_not_raise(),
           out='soilmstress_ar')),
     (dict(args=dict(soilm='soilm_mat', meanalpha='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_soilmstress(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_soilmstress(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_viscosity_h2o - temp + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc', patm='patm'),  # scalars
           cmng=does_not_raise(),
           out='viscosity_h2o_sc')),
     (dict(args=dict(tc='tc', patm='patm_mat'),  # scalar + array
           cmng=does_not_raise(),
           out='viscosity_h2o_sc_ar')),
     (dict(args=dict(tc='tc_mat', patm='patm_mat'),  # arrays
           cmng=does_not_raise(),
           out='viscosity_h2o_ar')),
     (dict(args=dict(tc='tc_mat', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_viscosity_h2o(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_viscosity_h2o(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_patm - elev only
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(elv='elev'),  # scalar
           cmng=does_not_raise(),
           out='patm_sc')),
     (dict(args=dict(elv='elev_mat'),  # array
           cmng=does_not_raise(),
           out='patm_ar'))]
)
def test_calc_patm(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_patm(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])

# ------------------------------------------
# Testing calc_co2_to_ca - co2 + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(co2='co2', patm='patm'),  # scalars
           cmng=does_not_raise(),
           out='co2_to_ca_sc')),
     (dict(args=dict(co2='co2_mat', patm='patm'),  # scalar + array
           cmng=does_not_raise(),
           out='co2_to_ca_sc_ar')),
     (dict(args=dict(co2='co2_mat', patm='patm_mat'),  # arrays
           cmng=does_not_raise(),
           out='co2_to_ca_ar')),
     (dict(args=dict(co2='co2_mat', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_co2_to_ca(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_co2_to_ca(**kwargs)
        assert np.allclose(ret, expected[ctrl['out']])


# ------------------------------------------
# Testing CalcOptimalChi - vpd + internals kmm, gammastar, ns_star, ca
#
# NOTE - the c4 method __INTENTIONALLY__ always returns scalars
#        regardless of input shape, so always uses the same expected values
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(kmm='kmm', gammastar='gammastar', ns_star='ns_star',
                     ca='ca', vpd='vpd'), method='c4',  # scalar, C4
           cmng=does_not_raise(),
           out='optchi_c4')),
     (dict(args=dict(kmm='kmm_mat', gammastar='gammastar', ns_star='ns_star_mat',
                     ca='ca', vpd='vpd'), method='c4',  # scalar + arrays, C4
           cmng=does_not_raise(),
           out='optchi_c4')),
     (dict(args=dict(kmm='kmm_mat', gammastar='gammastar_mat', ns_star='ns_star_mat',
                     ca='ca_mat', vpd='vpd_mat'), method='c4',  # scalar + arrays, C4
           cmng=does_not_raise(),
           out='optchi_c4')),
     (dict(args=dict(kmm='kmm_mat', gammastar='shape_error', ns_star='ns_star_mat',
                     ca='ca_mat', vpd='vpd_mat'), method='c4',  # scalar + arrays, C4
           cmng=pytest.raises(ValueError),
           out=None)),
     (dict(args=dict(kmm='kmm', gammastar='gammastar', ns_star='ns_star',
                     ca='ca', vpd='vpd'), method='prentice14',  # scalar, c3
           cmng=does_not_raise(),
           out='optchi_p14_sc')),
     (dict(args=dict(kmm='kmm_mat', gammastar='gammastar', ns_star='ns_star_mat',
                     ca='ca', vpd='vpd'), method='prentice14',  # scalar + arrays, c3
           cmng=does_not_raise(),
           out='optchi_p14_sc_ar')),
     (dict(args=dict(kmm='kmm_mat', gammastar='gammastar_mat', ns_star='ns_star_mat',
                     ca='ca_mat', vpd='vpd_mat'), method='prentice14',  # scalar + arrays, c3
           cmng=does_not_raise(),
           out='optchi_p14_ar')),
     (dict(args=dict(kmm='kmm_mat', gammastar='shape_error', ns_star='ns_star_mat',
                     ca='ca_mat', vpd='vpd_mat'), method='prentice14',  # scalar + arrays, c3
           cmng=pytest.raises(ValueError),
           out=None))
     ]
)
def test_calc_optimal_chi(inputs, expected, ctrl):

    with ctrl['cmng']:
        kwargs = {k: inputs[v] for k, v in ctrl['args'].items()}
        ret = pmodel.CalcOptimalChi(**kwargs, method=ctrl['method'])

        expected = expected[ctrl['out']]
        assert np.allclose(ret.chi, expected['chi'])
        assert np.allclose(ret.mj, expected['mj'])
        assert np.allclose(ret.mc, expected['mc'])
        assert np.allclose(ret.mjoc, expected['mjoc'])


# # ------------------------------------------
# # Testing CalcLUEVcmax- output of CalcOptimalChi + optional kphio,
# # ftemp_kphio and soilmstress. This has quite a few combinations,
# # depending on options to do soil moisture stress or kphio temperature
# # correction. The default kphio varies with settings but imposing a
# # single value here to keep test simpler.
# # - scalar vs array optchi inputs.
# # - four methods.
# # ------------------------------------------
#
#
# def test_calc_lue_vcmax_broadcast_failure(common_inputs):
#
#     with pytest.raises(ValueError):
#         oc = pmodel.CalcOptimalChi(common_inputs.kmm_mat,
#                                    common_inputs.gammastar_mat,
#                                    common_inputs.ns_star,
#                                    common_inputs.ca,
#                                    common_inputs.vpd)
#         _ = pmodel.CalcLUEVcmax(oc, common_inputs.kphio_broadcast_err)
#
#
#
#
# def test_calc_optimal_chi_c4_scalars(common_inputs):
#
#     ret = pmodel.CalcOptimalChi(common_inputs.kmm,
#                                 common_inputs.gammastar,
#                                 common_inputs.ns_star,
#                                 common_inputs.ca,
#                                 common_inputs.vpd,
#                                 method='c4')
#
#     assert ret.chi == 1.0
#     assert ret.mj == 1.0
#     assert ret.mjoc == 1.0
#     assert ret.mc == 1.0

