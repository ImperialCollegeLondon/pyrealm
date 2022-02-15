import os
import warnings
import pytest
import numpy as np
import json
from contextlib import contextmanager
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
    with open(os.path.join(test_dir, 'test_outputs_rpmodel.json')) as infile:
        values = json.load(infile)

    # JSON contains nested dictionary of scalars and lists - convert
    # the lists to ndarrays, and use float to ensure that None --> np.nan
    # rather than the default output of dtype object.

    def lists_to_ndarray(d):
        for k, v in d.items():
            if isinstance(v, dict):
                lists_to_ndarray(v) 
            elif  isinstance(v, list):
                d[k] = np.array(v, dtype=float)
            else:
                pass
    
    lists_to_ndarray(values)

    return values

# ------------------------------------------
# Test structure
# The basic structure of these tests  is to use a pytest.mark.parameterise
# to pass in the variables to be passed to the function along with the key
# for sets of expected ouputs and any context managers.
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
    'tc, patm, context_manager, expvals',
    [   ('tc_sc', 'patm_sc', does_not_raise(), 'dens_h20_sc'), # scalars
        ('tc_ar', 'patm_sc', does_not_raise(), 'dens_h20_mx'), # mixed
        ('tc_ar', 'patm_ar', does_not_raise(), 'dens_h20_ar'), # arrays
        ('tc_ar', 'shape_error', pytest.raises(ValueError), None), # shape mismatch
    ])
def test_calc_density_h2o(values, tc, patm, context_manager, expvals):

    with context_manager:

        ret = pmodel.calc_density_h2o(tc=values[tc], patm=values[patm])
        if expvals is not None:
            assert np.allclose(ret, values[expvals])

# ------------------------------------------
# Testing calc_ftemp_inst_rd - temp only but in Kelvin!
# ------------------------------------------

@pytest.mark.parametrize(
    'tk, expvars',
    [   ('tk_sc', 'ftemp_arrh_sc'),  # scalar
        ('tk_ar', 'ftemp_arrh_ar'),  # array
    ])
def test_calc_ftemp_arrh(values, tk, expvars):

    ret = pmodel.calc_ftemp_arrh(tk=values[tk], ha=values['KattgeKnorr_ha'])
    assert np.allclose(ret, values[expvars])

# ------------------------------------------
# Testing calc_ftemp_inst_vcmax - temp only
# ------------------------------------------

@pytest.mark.parametrize(
    'tc, expvars',
    [   ('tc_sc', 'ftemp_inst_vcmax_sc'), # scalar
        ('tc_ar', 'ftemp_inst_vcmax_ar'), # array
    ])
def test_calc_ftemp_inst_vcmax(values, tc, expvars):

    ret = pmodel.calc_ftemp_inst_vcmax(values[tc])
    assert np.allclose(ret, values[expvars])

# ------------------------------------------
# Testing calc_ftemp_inst_vcmax - temp only
# ------------------------------------------

# TODO - submit pull request to rpmodel with fix for this

@pytest.mark.parametrize(
    'tc, c4, expvars',
    [   ('tc_sc', False, 'ftemp_kphio_c3_sc'),  # scalar, C3
        ('tc_ar', False, 'ftemp_kphio_c3_ar'),  # array, C3
        ('tc_sc', True, 'ftemp_kphio_c4_sc'),  # scalar, C4
        ('tc_ar', True, 'ftemp_kphio_c4_ar'),  # array, C4
     ]
)
def test_calc_ftemp_kphio(values, tc, c4, expvars):

    ret = pmodel.calc_ftemp_kphio(tc=values[tc], c4=c4)
    assert np.allclose(ret, values[expvars])

# ------------------------------------------
# Testing calc_gammastar - temp + patm
# ------------------------------------------

@pytest.mark.parametrize(
    'tc, patm, context_manager, expvals',
    [   ('tc_sc', 'patm_sc', does_not_raise(), 'gammastar_sc'), # scalars
        ('tc_ar', 'patm_sc', does_not_raise(), 'gammastar_mx'), # mixed
        ('tc_ar', 'patm_ar', does_not_raise(), 'gammastar_ar'), # arrays
        ('tc_ar', 'shape_error', pytest.raises(ValueError), None), # shape mismatch
    ])
def test_calc_gammastar(values, tc, patm, context_manager, expvals):

    with context_manager:
        
        ret = pmodel.calc_gammastar(tc=values[tc], patm=values[patm])
        if expvals is not None:
            assert np.allclose(ret, values[expvals])

# ------------------------------------------
# Testing calc_kmm - temp + patm
# ------------------------------------------

@pytest.mark.parametrize(
    'tc, patm, context_manager, expvals',
    [   ('tc_sc', 'patm_sc', does_not_raise(), 'kmm_sc'), # scalars
        ('tc_ar', 'patm_sc', does_not_raise(), 'kmm_mx'), # mixed
        ('tc_ar', 'patm_ar', does_not_raise(), 'kmm_ar'), # arrays
        ('tc_ar', 'shape_error', pytest.raises(ValueError), None), # shape mismatch
    ]
)
def test_calc_kmm(values, tc, patm, context_manager, expvals):

    with context_manager:
        
        ret = pmodel.calc_kmm(tc=values[tc], patm=values[patm])
        if expvals:
            assert np.allclose(ret, values[expvals])

# ------------------------------------------
# Testing calc_soilmstress - soilm + meanalpha
# ------------------------------------------

@pytest.mark.parametrize(
    'soilm, meanalpha, context_manager, expvals',
    [   ('soilm_sc', 'meanalpha_sc', does_not_raise(), 'soilmstress_sc'), # scalars
        ('soilm_ar', 'meanalpha_sc', does_not_raise(), 'soilmstress_mx'), # mixed
        ('soilm_ar', 'meanalpha_ar', does_not_raise(), 'soilmstress_ar'), # arrays
        ('soilm_ar', 'shape_error', pytest.raises(ValueError), None), # shape mismatch
    ]
)
def test_calc_soilmstress(values, soilm, meanalpha, context_manager, expvals):

    with context_manager:
        
        ret = pmodel.calc_soilmstress(soilm=values[soilm], meanalpha=values[meanalpha])
        if expvals:
            assert np.allclose(ret, values[expvals])

# ------------------------------------------
# Testing calc_viscosity_h2o - temp + patm
# ------------------------------------------


@pytest.mark.parametrize(
    'tc, patm, context_manager, expvals',
    [   ('tc_sc', 'patm_sc', does_not_raise(), 'viscosity_h2o_sc'), # scalars
        ('tc_ar', 'patm_sc', does_not_raise(), 'viscosity_h2o_mx'), # mixed
        ('tc_ar', 'patm_ar', does_not_raise(), 'viscosity_h2o_ar'), # arrays
        ('tc_ar', 'shape_error', pytest.raises(ValueError), None), # shape mismatch
    ]
)
def test_calc_viscosity_h2o(values, tc, patm, context_manager, expvals):

   with context_manager:
        
        ret = pmodel.calc_viscosity_h2o(tc=values[tc], patm=values[patm])
        if expvals:
            assert np.allclose(ret, values[expvals])

# ------------------------------------------
# Testing calc_patm - elev only
# ------------------------------------------

@pytest.mark.parametrize(
    'elev, expvals',
    [   ('elev_sc', 'patm_from_elev_sc'),  # scalars
        ('elev_ar', 'patm_from_elev_ar'),  # arrays
    ]
)
def test_calc_patm(values, elev, expvals):

    ret = pmodel.calc_patm(elv=values[elev])
    assert np.allclose(ret, values[expvals])

# ------------------------------------------
# Testing calc_co2_to_ca - co2 + patm
# ------------------------------------------

@pytest.mark.parametrize(
    'co2, patm, context_manager, expvals',
    [   ('co2_sc', 'patm_sc', does_not_raise(), 'ca_sc'), # scalars
        ('co2_ar', 'patm_sc', does_not_raise(), 'ca_mx'), # mixed
        ('co2_ar', 'patm_ar', does_not_raise(), 'ca_ar'), # arrays
        ('co2_ar', 'shape_error', pytest.raises(ValueError), None), # shape mismatch
    ]
)
def test_calc_co2_to_ca(values, co2, patm, context_manager, expvals):

    with context_manager:
        
        ret = pmodel.calc_co2_to_ca(co2=values[co2], patm=values[patm])
        if expvals:
            assert np.allclose(ret, values[expvals])

# ------------------------------------------
# Testing CalcOptimalChi - vpd + internals kmm, gammastar, ns_star, ca
# ------------------------------------------

@pytest.mark.parametrize(
    'tc, patm, co2, vpd, method, context_manager, expvalues',
    [ 
        ('tc_sc', 'patm_sc', 'co2_sc', 'vpd_sc',
         'c4', does_not_raise(), 'optchi_p14_sc_c4'),  # scalar, c4
        ('tc_ar', 'patm_sc', 'co2_sc', 'vpd_sc',
         'c4', does_not_raise(), 'optchi_p14_mx_c4'),  # scalar + arrays, c4
        ('tc_ar', 'patm_ar', 'co2_ar', 'vpd_ar',
         'c4', does_not_raise(), 'optchi_p14_ar_c4'),  # arrays, c4
        ('shape_error', 'patm_ar', 'co2_ar', 'vpd_ar',
         'c4', pytest.raises(ValueError), None),  # shape error, c4
        ('tc_sc', 'patm_sc', 'co2_sc', 'vpd_sc',
         'prentice14', does_not_raise(), 'optchi_p14_sc_c3'),  # scalar, c3
        ('tc_ar', 'patm_sc', 'co2_sc', 'vpd_sc',
         'prentice14', does_not_raise(), 'optchi_p14_mx_c3'),  # scalar + arrays, c3
        ('tc_ar', 'patm_ar', 'co2_ar', 'vpd_ar',
         'prentice14', does_not_raise(), 'optchi_p14_ar_c3'),  # arrays, c3
        ('shape_error', 'patm_ar', 'co2_ar', 'vpd_ar',
         'prentice14', pytest.raises(ValueError), None),  # shape error, c3
     ]
)
def test_calc_optimal_chi(values, tc, patm, co2, vpd, method, context_manager, expvalues):

    with context_manager:

        env = pmodel.PModelEnvironment(tc=values[tc],
                                       patm=values[patm],
                                       vpd=values[vpd],
                                       co2=values[co2])

        ret = pmodel.CalcOptimalChi(env, method=method)

        if expvalues is not None:
            expected = values[expvalues]
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
    'soilm, meanalpha',
    [(None, None),
     ('soilm_sc', 'meanalpha_sc')],
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
    'tc, patm, co2, vpd',
    [ 
        ('tc_sc', 'patm_sc', 'co2_sc', 'vpd_sc'),  # scalar
        ('tc_ar', 'patm_ar', 'co2_ar', 'vpd_ar'),  # arrays
     ],
    ids = ['sc', 'ar']
)
@pytest.mark.parametrize(
    'c4',
    [True, False],
    ids=['c4', 'c3']
)
def test_calc_lue_vcmax(request, values, soilm, meanalpha, ftemp_kphio,
                        luevcmax_method, tc, patm, co2, vpd, c4):

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
    elif tc == 'tc_sc':
        ftemp_kphio = pmodel.calc_ftemp_kphio(tc=values[tc], c4=c4)
    else:
        ftemp_kphio = pmodel.calc_ftemp_kphio(tc=values[tc], c4=c4)

    # Optimal Chi
    env = pmodel.PModelEnvironment(tc=values[tc],
                                   patm=values[patm],
                                   vpd=values[vpd],
                                   co2=values[co2])

    optchi = pmodel.CalcOptimalChi(env, method=oc_method)

    # Soilmstress
    if soilm is None:
        soilmstress = 1.0
    else:
        soilmstress = pmodel.calc_soilmstress(soilm=values[soilm],
                                              meanalpha=values[meanalpha])

    ret = pmodel.CalcLUEVcmax(optchi, kphio=0.05, ftemp_kphio=ftemp_kphio,
                              soilmstress=soilmstress, method=luevcmax_method)

    # Find the expected values, extracting the combination from the request
    name = request.node.name
    name = name[(name.find('[') + 1):-1]
    expected = values['jmax-' + name]

    assert np.allclose(ret.lue, expected['lue'], equal_nan=True)
    assert np.allclose(ret.vcmax, expected['vcmax_unitiabs'], equal_nan=True)

    if luevcmax_method == 'smith19':
        assert np.allclose(ret.omega, expected['omega'], equal_nan=True)
        assert np.allclose(ret.omega_star, expected['omega_star'], equal_nan=True)

# ------------------------------------------
# Testing PModelEnvironment class
# - double checking that the bundled calcs in here work as expected
# - test that the constraint issues a warning as expected.
# ------------------------------------------

@pytest.mark.parametrize(
    'tc, vpd, co2, patm, ca, kmm, gammastar, ns_star',
    [('tc_sc', 'vpd_sc', 'co2_sc', 'patm_sc', 'ca_sc', 'kmm_sc', 'gammastar_sc', 'ns_star_sc'),
     ('tc_ar', 'vpd_ar', 'co2_ar', 'patm_ar', 'ca_ar', 'kmm_ar', 'gammastar_ar', 'ns_star_ar')],
    ids=['sc', 'ar']
)
def test_pmodelenvironment(values, tc, vpd, co2, patm, ca, kmm, gammastar, ns_star):

    ret = pmodel.PModelEnvironment(tc=values[tc],
                                   patm=values[patm],
                                   vpd=values[vpd],
                                   co2=values[co2]
                                   )

    assert np.allclose(ret.gammastar, values[gammastar])
    assert np.allclose(ret.ns_star, values[ns_star])
    assert np.allclose(ret.kmm, values[kmm])
    assert np.allclose(ret.ca, values[ca])


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
