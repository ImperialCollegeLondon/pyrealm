import os
import warnings
import pytest
import numpy as np
import yaml
from contextlib import contextmanager
# import pkg_resources
from pyrealm import pmodel


# RPMODEL bugs
# rpmodel was using an incorrect parameterisation of the C4 ftemp kphio curve
# that is fixed but currently (1.2.0) an implementation error in the output checking 
# means this still have to be skipped.

RPMODEL_C4_BUG = True

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

    test_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(test_dir, 'test_outputs_rpmodel.yaml')) as infile:
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
     (dict(args=dict(tc='tc_ar', patm='patm_sc'),  # scalar + array
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
    [(dict(args=dict(tk='tk_sc', ha='KattgeKnorr_ha'),  # scalar
           cmng=does_not_raise(),
           out='ftemp_arrh_sc')),
     (dict(args=dict(tk='tk_ar', ha='KattgeKnorr_ha'),  # array
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


@pytest.mark.skipif(RPMODEL_C4_BUG, reason='Benchmark incorrect')
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
     (dict(args=dict(tc='tc_ar', patm='patm_sc'),  # scalar + array
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

    if ctrl['method'] == 'c4':
        pytest.skip('Not currently testing C4 outputs - chi estimates for C4 need estimating as C3.')


    with ctrl['cmng']:
        kwargs = {k: values[v] for k, v in ctrl['args'].items()}
        ret = pmodel.CalcOptimalChi(**kwargs, method=ctrl['method'])

        expected = values[ctrl['out']]
        assert np.allclose(ret.chi, expected['chi'])
        assert np.allclose(ret.mj, expected['mj'])
        assert np.allclose(ret.mc, expected['mc'])
        assert np.allclose(ret.mjoc, expected['mjoc'])

# ------------------------------------------
# Testing CalcLUEVcmax -  This has quite a few combinations:
# - c4
# - optchi - using both scalar and array inputs.
# - soilmstress - only considering a scalar input here (so a uniform soil stress
#       across array optchi variants) rather than creating tests that have a
#       scalar optchi applied across array of soil moistures.
# - ftemp_kphio
# - CalcLUEVcmax method
# - kphio - normally this varies with the pmodel input setup but imposing a
#       single value here (0.05) to simplify test suite.
# ------------------------------------------


@pytest.mark.parametrize(
    'soilmstress',
    [dict(soilm=None, meanalpha=None),
     dict(soilm='soilm_sc', meanalpha='meanalpha_sc')],
    ids=['sm-off', 'sm-on']
)
@pytest.mark.parametrize(
    'ftemp_kphio',
    [True, False],
    ids=['fkphio-on', 'fkphio-off']
)
@pytest.mark.parametrize(
    'luevcmax_method',
    ['wang17', 'smith19', 'none'],
    ids=['wang17', 'smith19', 'none']
)
@pytest.mark.parametrize(
    'optchi',
    [dict(kmm='kmm_sc', gammastar='gammastar_sc',
          ns_star='ns_star_sc', ca='ca_sc', vpd='vpd_sc'),
     dict(kmm='kmm_ar', gammastar='gammastar_ar',
          ns_star='ns_star_ar', ca='ca_ar', vpd='vpd_ar')],
    ids=['sc', 'ar']
)
@pytest.mark.parametrize(
    'c4',
    [True],  # , False],
    ids=['c4']  # , 'c3']
)
def test_calc_lue_vcmax(request, values, soilmstress, ftemp_kphio,
                        luevcmax_method, optchi, c4):

    # ftemp_kphio needs to know the original tc inputs to optchi - these have
    # all been synchronised so that anything with type 'mx' or 'ar' used the
    # tc_ar input

    if c4:
        oc_method = 'c4'
        pytest.skip('Not currently testing C4 outputs because of param in rpmodel')
    else:
        oc_method = 'prentice14'

    if not ftemp_kphio:
        ftemp_kphio = 1.0
    elif optchi['kmm'] == 'kmm_sc':
        ftemp_kphio = pmodel.calc_ftemp_kphio(tc=values['tc_sc'], c4=c4)
    else:
        ftemp_kphio = pmodel.calc_ftemp_kphio(tc=values['tc_ar'], c4=c4)

    # Optimal Chi
    kwargs = {k: values[v] for k, v in optchi.items()}
    optchi = pmodel.CalcOptimalChi(**kwargs, method=oc_method)

    # Soilmstress
    if soilmstress['soilm'] is None:
        soilmstress = 1.0
    else:
        soilmstress = pmodel.calc_soilmstress(soilm=values[soilmstress['soilm']],
                                              meanalpha=values[soilmstress['meanalpha']])

    ret = pmodel.CalcLUEVcmax(optchi, kphio=0.05, ftemp_kphio=ftemp_kphio,
                              soilmstress=soilmstress, method=luevcmax_method)

    # Find the expected values, extracting the combination from the request
    name = request.node.name
    name = name[(name.find('[') + 1):-1]
    expected = values['jmax-' + name]

    assert np.allclose(ret.lue, expected['lue'])
    assert np.allclose(ret.vcmax, expected['vcmax_unitiabs'])

    if luevcmax_method == 'smith19':
        assert np.allclose(ret.omega, expected['omega'])
        assert np.allclose(ret.omega_star, expected['omega_star'])

# ------------------------------------------
# Testing PModelEnvironment class
# - double checking that the bundled calcs in here work as expected
# - test that the constraint issues a warning as expected.
# ------------------------------------------

@pytest.mark.parametrize(
    'variables',
    [dict(tc='tc_sc', vpd='vpd_sc', co2='co2_sc', patm='patm_sc',
          ca='ca_sc', kmm='kmm_sc', gammastar='gammastar_sc', ns_star='ns_star_sc'),
     dict(tc='tc_ar', vpd='vpd_ar', co2='co2_ar', patm='patm_ar',
          ca='ca_ar', kmm='kmm_ar', gammastar='gammastar_ar', ns_star='ns_star_ar')],
    ids=['sc', 'ar']
)
def test_pmodelenvironment(values, variables):

    ret = pmodel.PModelEnvironment(tc=values[variables['tc']],
                                   patm=values[variables['patm']],
                                   vpd=values[variables['vpd']],
                                   co2=values[variables['co2']]
                                   )

    assert np.allclose(ret.gammastar, values[variables['gammastar']])
    assert np.allclose(ret.ns_star, values[variables['ns_star']])
    assert np.allclose(ret.kmm, values[variables['kmm']])
    assert np.allclose(ret.ca, values[variables['ca']])


def test_pmodelenvironment_constraint():

    with pytest.warns(UserWarning):
        ret = pmodel.PModelEnvironment(tc=np.array([-15, 5, 10, 15, 20]),
                                       vpd=100000,
                                       co2=400,
                                       patm=101325)


def test_pmodelenvironment_toocold():

    with pytest.raises(ValueError):
        ret = pmodel.PModelEnvironment(tc=np.array([-35, 5, 10, 15, 20]),
                                       vpd=1000,
                                       co2=400,
                                       patm=101325)


def test_pmodelenvironment_dewpoint():

    with pytest.raises(ValueError):
        ret = pmodel.PModelEnvironment(tc=np.array([-15, 5, 10, 15, 20]),
                                       vpd=-1,
                                       co2=400,
                                       patm=101325)


# ------------------------------------------
# Testing PModel class - separate c3 and c4 tests
# - sc + ar inputs: tc, vpd, co2, patm (not testing elev)
# - +- soilmstress: soilm, meanalpha (but again assuming constant across ar inputs)
# - do_ftemp_kphio
# - luevcmax method

# For all tests
# - include fapar_sc and ppfd_sc (same irradiation everywhere)
# - hold kphio static
# ------------------------------------------


@pytest.fixture(scope='module')
def pmodelenv(values):
    """Fixture to create PModelEnvironments with scalar and array inputs
    """

    sc = pmodel.PModelEnvironment(tc=values['tc_sc'],
                                  vpd=values['vpd_sc'],
                                  co2=values['co2_sc'],
                                  patm=values['patm_sc'])

    ar = pmodel.PModelEnvironment(tc=values['tc_ar'],
                                  vpd=values['vpd_ar'],
                                  co2=values['co2_ar'],
                                  patm=values['patm_ar'])

    return {'sc': sc, 'ar': ar}


@pytest.mark.parametrize(
    'soilmstress',
    [False, True],
    ids=['sm-off', 'sm-on']
)
@pytest.mark.parametrize(
    'ftemp_kphio',
    [True, False],
    ids=['fkphio-on', 'fkphio-off']
)
@pytest.mark.parametrize(
    'luevcmax_method',
    ['wang17', 'smith19', 'none'],
    ids=['wang17', 'smith19', 'none']
)
@pytest.mark.parametrize(
    'environ',
    ['sc', 'ar'],
    ids=['sc', 'ar']
)
def test_pmodel_class_c3(request, values, pmodelenv, soilmstress, ftemp_kphio, luevcmax_method, environ):


    if soilmstress:
        soilmstress = pmodel.calc_soilmstress(values['soilm_sc'], values['meanalpha_sc'])
    else:
        soilmstress = None

    ret = pmodel.PModel(pmodelenv[environ],
                        kphio=0.05,
                        soilmstress=soilmstress,
                        do_ftemp_kphio=ftemp_kphio,
                        method_jmaxlim=luevcmax_method)

    # Find the expected values, extracting the combination from the request
    name = request.node.name
    name = name[(name.find('[') + 1):-1]
    expected = values['rpmodel-c3-' + name + '-unitiabs']

    # Test values - two values calculated in main rpmodel function
    # so can only test here - ci and iwue
    assert np.allclose(ret.iwue, expected['iwue'])
    assert np.allclose(ret.optchi.ci, expected['ci'])

    # - and six values that are scaled by IABS - rpmodel enforces scaling
    # where PModel can do it post hoc from unit_iabs values, so two
    # rpmodel runs are used to test the unit values and scaled.
    assert np.allclose(ret.unit_iabs.lue, expected['lue'])
    assert np.allclose(ret.unit_iabs.vcmax, expected['vcmax'])
    assert np.allclose(ret.unit_iabs.vcmax25, expected['vcmax25'])
    assert np.allclose(ret.unit_iabs.rd, expected['rd'])
    assert np.allclose(ret.unit_iabs.gs, expected['gs'])

    # TODO - Numerical instability in the Jmax calculation - as denominator
    #        approaches 1, results --> infinity unpredictably with rounding
    #        so currently excluding Jmax in combinations where this occurs.

    if 'none-fkphio-off-sm-off' not in name:
        assert np.allclose(ret.unit_iabs.jmax, expected['jmax'])
    else:
        warnings.warn('Skipping Jmax test for cases with numerical instability')

    # Check Iabs scaling
    iabs = ret.unit_iabs.scale_iabs(values['fapar_sc'], values['ppfd_sc'])

    # Find the expected values, extracting the combination from the request
    expected = values['rpmodel-c3-' + name + '-iabs']

    assert np.allclose(iabs.gpp, expected['gpp'])
    assert np.allclose(iabs.vcmax, expected['vcmax'])
    assert np.allclose(iabs.vcmax25, expected['vcmax25'])
    assert np.allclose(iabs.rd, expected['rd'])
    assert np.allclose(iabs.gs, expected['gs'])

    if 'none-fkphio-off-sm-off' not in name:
        assert np.allclose(iabs.jmax, expected['jmax'])
    else:
        warnings.warn('Skipping Jmax test for cases with numerical instability')


# Testing PModel class with C4

@pytest.mark.skipif(RPMODEL_C4_BUG, reason='Benchmark incorrect')
@pytest.mark.parametrize(
    'soilmstress',
    [False, True],
    ids=['sm-off', 'sm-on']
)
@pytest.mark.parametrize(
    'ftemp_kphio',
    [True, False],
    ids=['fkphio-on', 'fkphio-off']
)
@pytest.mark.parametrize(
    'environ',
    ['sc', 'ar'],
    ids=['sc', 'ar']
)
def test_pmodel_class_c3(request, values, pmodelenv, soilmstress, ftemp_kphio, environ):

    if soilmstress:
        soilmstress = pmodel.calc_soilmstress(values['soilm_sc'], values['meanalpha_sc'])
    else:
        soilmstress = None

    ret = pmodel.PModel(pmodelenv[environ],
                        kphio=0.05,
                        soilmstress=soilmstress,
                        do_ftemp_kphio=ftemp_kphio,
                        method_jmaxlim='none',  # enforced in rpmodel.
                        c4=True)

    # Find the expected values, extracting the combination from the request
    name = request.node.name
    name = name[(name.find('[') + 1):-1]
    expected = values['rpmodel-c4-' + name + '-unitiabs']

    # Test values - two values calculated in main rpmodel function
    # so can only test here - ci and iwue
    assert np.allclose(ret.iwue, expected['iwue'])
    assert np.allclose(ret.optchi.ci, expected['ci'])

    # - and six values that are scaled by IABS - rpmodel enforces scaling
    # where PModel can do it post hoc from unit_iabs values, so two
    # rpmodel runs are used to test the unit values and scaled.

    ret.estimate_productivity()  # defaults of fapar=1, ppfd=1

    assert np.allclose(ret.lue, expected['lue'])
    assert np.allclose(ret.vcmax, expected['vcmax'])
    assert np.allclose(ret.vcmax25, expected['vcmax25'])
    assert np.allclose(ret.rd, expected['rd'])
    assert np.allclose(ret.jmax, expected['jmax'])
    assert np.allclose(ret.gs, expected['gs'])

    # Check Iabs scaling
    ret.estimate_productivity(fapar=values['fapar_sc'],
                              ppfd=values['ppfd_sc'])

    # Find the expected values, extracting the combination from the request
    expected = values['rpmodel-c4-' + name + '-iabs']

    assert np.allclose(ret.gpp, expected['gpp'])
    assert np.allclose(ret.vcmax, expected['vcmax'])
    assert np.allclose(ret.vcmax25, expected['vcmax25'])
    assert np.allclose(ret.rd, expected['rd'])
    assert np.allclose(ret.jmax, expected['jmax'])
    assert np.allclose(ret.gs, expected['gs'])
