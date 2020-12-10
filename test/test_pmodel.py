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
def values():
    """Fixture to load test inputs from file.
    """

    with open('test/test_outputs_rpmodel.yaml') as infile:
        values = yaml.load(infile, Loader=yaml.SafeLoader)

    values = {k: np.array(v) if isinstance(v, list) else v
              for k, v in values.items()}

    return values

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
    [(dict(args=dict(tc='tc_sc', patm='patm_sc'),  # scalars
           cmng=does_not_raise(),
           out='dens_h20_sc')),
     (dict(args=dict(tc='tc_sc', patm='patm_ar'),  # scalar + array
           cmng=does_not_raise(),
           out='dens_h20_mx')),
     (dict(args=dict(tc='tc_ar', patm='patm_ar'),  # arrays
           cmng=does_not_raise(),
           out='dens_h20_ar')),
     (dict(args=dict(tc='tc_ar', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_density_h2o(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_density_h2o(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_ftemp_inst_rd - temp only but in Kelvin!
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tk='tk_sc', dha='KattgeKnorr_ha'),  # scalar
           cmng=does_not_raise(),
           out='ftemp_arrh_sc')),
     (dict(args=dict(tk='tk_ar', dha='KattgeKnorr_ha'),  # array
           cmng=does_not_raise(),
           out='ftemp_arrh_ar'))]
)
def test_calc_ftemp_arrh(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_ftemp_arrh(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_ftemp_inst_vcmax - temp only
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc_sc'),  # scalar
           cmng=does_not_raise(),
           out='ftemp_inst_vcmax_sc')),
     (dict(args=dict(tc='tc_ar'),  # array
           cmng=does_not_raise(),
           out='ftemp_inst_vcmax_ar'))]
)
def test_calc_ftemp_inst_vcmax(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_ftemp_inst_vcmax(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_ftemp_inst_vcmax - temp only
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc_sc'), c4=False,  # scalar, C3
           cmng=does_not_raise(),
           out='ftemp_kphio_c3_sc')),
     (dict(args=dict(tc='tc_ar'), c4=False,  # array, C3
           cmng=does_not_raise(),
           out='ftemp_kphio_c3_ar')),
     (dict(args=dict(tc='tc_sc'), c4=True,  # scalar, C4
           cmng=does_not_raise(),
           out='ftemp_kphio_c4_sc')),
     (dict(args=dict(tc='tc_ar'), c4=True,  # array, C4
           cmng=does_not_raise(),
           out='ftemp_kphio_c4_ar'))]
)
def test_calc_ftemp_kphio(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_ftemp_kphio(**kwargs, c4=ctrl['c4'])
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_gammastar - temp + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc_sc', patm='patm_sc'),  # scalars
           cmng=does_not_raise(),
           out='gammastar_sc')),
     (dict(args=dict(tc='tc_ar', patm='patm_sc'),  # scalar + array
           cmng=does_not_raise(),
           out='gammastar_mx')),
     (dict(args=dict(tc='tc_ar', patm='patm_ar'),  # arrays
           cmng=does_not_raise(),
           out='gammastar_ar')),
     (dict(args=dict(tc='tc_ar', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_gammastar(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_gammastar(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_kmm - temp + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc_sc', patm='patm_sc'),  # scalars
           cmng=does_not_raise(),
           out='kmm_sc')),
     (dict(args=dict(tc='tc_ar', patm='patm_sc'),  # scalar + array
           cmng=does_not_raise(),
           out='kmm_mx')),
     (dict(args=dict(tc='tc_ar', patm='patm_ar'),  # arrays
           cmng=does_not_raise(),
           out='kmm_ar')),
     (dict(args=dict(tc='tc_ar', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_kmm(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_kmm(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_soilmstress - soilm + meanalpha
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(soilm='soilm_sc', meanalpha='meanalpha_sc'),  # scalars
           cmng=does_not_raise(),
           out='soilmstress_sc')),
     (dict(args=dict(soilm='soilm_ar', meanalpha='meanalpha_sc'),  # scalar + array
           cmng=does_not_raise(),
           out='soilmstress_mx')),
     (dict(args=dict(soilm='soilm_ar', meanalpha='meanalpha_ar'),  # arrays
           cmng=does_not_raise(),
           out='soilmstress_ar')),
     (dict(args=dict(soilm='soilm_ar', meanalpha='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_soilmstress(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_soilmstress(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_viscosity_h2o - temp + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(tc='tc_sc', patm='patm_sc'),  # scalars
           cmng=does_not_raise(),
           out='viscosity_h2o_sc')),
     (dict(args=dict(tc='tc_sc', patm='patm_ar'),  # scalar + array
           cmng=does_not_raise(),
           out='viscosity_h2o_mx')),
     (dict(args=dict(tc='tc_ar', patm='patm_ar'),  # arrays
           cmng=does_not_raise(),
           out='viscosity_h2o_ar')),
     (dict(args=dict(tc='tc_ar', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_viscosity_h2o(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_viscosity_h2o(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_patm - elev only
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(elv='elev_sc'),  # scalar
           cmng=does_not_raise(),
           out='patm_from_elev_sc')),
     (dict(args=dict(elv='elev_ar'),  # array
           cmng=does_not_raise(),
           out='patm_from_elev_ar'))]
)
def test_calc_patm(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_patm(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])

# ------------------------------------------
# Testing calc_co2_to_ca - co2 + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(co2='co2_sc', patm='patm_sc'),  # scalars
           cmng=does_not_raise(),
           out='ca_sc')),
     (dict(args=dict(co2='co2_ar', patm='patm_sc'),  # scalar + array
           cmng=does_not_raise(),
           out='ca_mx')),
     (dict(args=dict(co2='co2_ar', patm='patm_ar'),  # arrays
           cmng=does_not_raise(),
           out='ca_ar')),
     (dict(args=dict(co2='co2_ar', patm='shape_error'),  # shape mismatch
           cmng=pytest.raises(ValueError),
           out=None))]
)
def test_calc_co2_to_ca(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.calc_co2_to_ca(**kwargs)
        assert np.allclose(ret, values[ctrl['out']])


# ------------------------------------------
# Testing CalcOptimalChi - vpd + internals kmm, gammastar, ns_star, ca
#
# NOTE - the c4 method __INTENTIONALLY__ always returns scalars
#        regardless of input shape, so always uses the same expected values
# ------------------------------------------


@pytest.mark.parametrize(
    'ctrl',
    [(dict(args=dict(kmm='kmm_sc', gammastar='gammastar_sc',
                     ns_star='ns_star_sc', ca='ca_sc', vpd='vpd_sc'),
           method='c4',  # scalar, C4
           cmng=does_not_raise(),
           out='optchi_c4')),
     (dict(args=dict(kmm='kmm_ar', gammastar='gammastar_sc',
                     ns_star='ns_star_ar', ca='ca_sc', vpd='vpd_sc'),
           method='c4',  # scalar + arrays, C4
           cmng=does_not_raise(),
           out='optchi_c4')),
     (dict(args=dict(kmm='kmm_ar', gammastar='gammastar_ar',
                     ns_star='ns_star_ar', ca='ca_ar', vpd='vpd_ar'),
           method='c4',  # scalar + arrays, C4
           cmng=does_not_raise(),
           out='optchi_c4')),
     (dict(args=dict(kmm='kmm_ar', gammastar='shape_error',
                     ns_star='ns_star_ar', ca='ca_ar', vpd='vpd_ar'),
           method='c4',  # scalar + arrays, C4
           cmng=pytest.raises(ValueError),
           out=None)),
     (dict(args=dict(kmm='kmm_sc', gammastar='gammastar_sc',
                     ns_star='ns_star_sc', ca='ca_sc', vpd='vpd_sc'),
           method='prentice14',  # scalar, c3
           cmng=does_not_raise(),
           out='optchi_p14_sc')),
     (dict(args=dict(kmm='kmm_ar', gammastar='gammastar_sc',
                     ns_star='ns_star_ar', ca='ca_sc', vpd='vpd_sc'),
           method='prentice14',  # scalar + arrays, c3
           cmng=does_not_raise(),
           out='optchi_p14_mx')),
     (dict(args=dict(kmm='kmm_ar', gammastar='gammastar_ar',
                     ns_star='ns_star_ar', ca='ca_ar', vpd='vpd_ar'),
           method='prentice14',  # scalar + arrays, c3
           cmng=does_not_raise(),
           out='optchi_p14_ar')),
     (dict(args=dict(kmm='kmm_ar', gammastar='shape_error',
                     ns_star='ns_star_ar', ca='ca_ar', vpd='vpd_ar'),
           method='prentice14',  # scalar + arrays, c3
           cmng=pytest.raises(ValueError),
           out=None))
     ]
)
def test_calc_optimal_chi(values, ctrl):

    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.CalcOptimalChi(**kwargs, method=ctrl['method'])

        expected = values[ctrl['out']]
        assert np.allclose(ret.chi, expected['chi'])
        assert np.allclose(ret.mj, expected['mj'])
        assert np.allclose(ret.mc, expected['mc'])
        assert np.allclose(ret.mjoc, expected['mjoc'])


# # ------------------------------------------
# # Testing CalcLUEVcmax -  This has quite a few combinations:
# # - c4
# # - soilmstress
# # - ftemp_kphio
# # - scalar vs array optchi
# # - method
# # - kphio also varies with input setup but imposing a single value
# #   here (0.05) to simplify test suite.
# # ------------------------------------------
#
#
# @pytest.mark.parametrize(
#     'soilmstress',
#     [dict(soilm=None, meanalpha=None),
#      dict(soilm='soilm_sc', meanalpha='meanalpha_sc'),
#      dict(soilm='soilm_ar', meanalpha='meanalpha_ar')]
# )
# @pytest.mark.parametrize(
#     'ftemp_kphio',
#     [True, False]
# )
# @pytest.mark.parametrize(
#     'luevcmax_method',
#     ['wang17', 'smith19', 'none']
# )
# @pytest.mark.parametrize(
#     'optchi',
#     [dict(args=dict(kmm='kmm_sc', gammastar='gammastar_sc',
#                     ns_star='ns_star_sc', ca='ca_sc', vpd='vpd_sc'),
#           type='sc'),
#      dict(args=dict(kmm='kmm_ar', gammastar='gammastar_sc',
#                     ns_star='ns_star_ar', ca='ca_sc', vpd='vpd_sc'),
#           type='mx'),
#      dict(args=dict(kmm='kmm_ar', gammastar='gammastar_ar',
#                     ns_star='ns_star_ar', ca='ca_ar', vpd='vpd_ar'),
#           type='ar')
#      ]
# )
# def test_calc_lue_vcmax_c3(values, soilmstress,
#                            ftemp_kphio, luevcmax_method, optchi):
#
#
#
#     expected_key = (f"lue_vcmax_{soilmstress['soilm']}_{ftemp_kphio}_" +
#                     f"{luevcmax_method}_{optchi['type']}_{optchi['method']}")
#
#     # Optimal Chi
#     kwargs = {k: values[v] for k, v in optchi['args'].items()}
#     optchi = pmodel.CalcOptimalChi(**kwargs, method=optchi['method'])
#
#     ftemp_kphio = pmodel.calc_ftemp_kphio(tc=) if ftemp_kphio else 1.0
#
#     soilmstress = pmodel.calc_soilmstress(soilm, meanalpha=) if soilm is not None else 1.0
#
#     ret = pmodel.CalcLUEVcmax(optchi, kphio=0.05, ftemp_kphio=ftemp_kphio,
#                               soilmstress=soilmstress,
#                               )
