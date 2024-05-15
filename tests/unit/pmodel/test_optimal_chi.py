"""Testing Optimal submodule."""

import numpy as np
import pytest

from pyrealm.pmodel.optimal_chi import (
    OptimalChiC4,
    OptimalChiC4NoGamma,
    OptimalChiC4NoGammaRootzoneStress,
    OptimalChiC4RootzoneStress,
    OptimalChiLavergne20C3,
    OptimalChiLavergne20C4,
    OptimalChiPrentice14,
    OptimalChiPrentice14RootzoneStress,
)
from pyrealm.pmodel.pmodel_environment import PModelEnvironment


@pytest.fixture
def photo_env():
    """Photosynthesis Environment setup."""
    return PModelEnvironment(
        tc=np.array([20]),
        vpd=np.array([1000]),
        co2=np.array([400]),
        patm=np.array([101325.0]),
        rootzonestress=np.array([1]),
        theta=np.array([0.5]),
    )


@pytest.mark.parametrize(
    """optimal_chi_class""",
    [
        OptimalChiPrentice14,
        OptimalChiPrentice14RootzoneStress,
        OptimalChiC4NoGammaRootzoneStress,
        OptimalChiC4,
        OptimalChiC4RootzoneStress,
        OptimalChiLavergne20C3,
        OptimalChiLavergne20C4,
        OptimalChiC4NoGamma,
    ],
)
def test_set_beta(optimal_chi_class, photo_env):
    """Test that beta is set correctly."""
    optimal_chi_instance = optimal_chi_class(env=photo_env)
    optimal_chi_instance.set_beta()
    # Test that beta attribute is set correctly
    assert isinstance(optimal_chi_instance.beta, np.ndarray)


@pytest.mark.parametrize(
    """optimal_chi_class""",
    [
        OptimalChiPrentice14,
        OptimalChiPrentice14RootzoneStress,
        OptimalChiC4NoGammaRootzoneStress,
        OptimalChiC4,
        OptimalChiC4RootzoneStress,
        OptimalChiLavergne20C3,
        OptimalChiLavergne20C4,
        OptimalChiC4NoGamma,
    ],
)
def test_estimate_chi(optimal_chi_class, photo_env):
    """Test that chi is estimated correctly."""
    optimal_chi_instance = optimal_chi_class(env=photo_env)
    optimal_chi_instance.set_beta()
    optimal_chi_instance.estimate_chi()
    # Test that chi and other related attributes are calculated correctly
    assert isinstance(optimal_chi_instance.chi, np.ndarray)
    assert isinstance(optimal_chi_instance.mc, np.ndarray)
    assert isinstance(optimal_chi_instance.mj, np.ndarray)
    assert isinstance(optimal_chi_instance.mjoc, np.ndarray)


@pytest.mark.parametrize(
    argnames=["subclass", "pmodelenv_args", "expected"],
    argvalues=[
        (
            OptimalChiPrentice14,
            dict(tc=20, patm=101325, co2=400, vpd=1000),
            dict(chi=0.69435, mc=0.33408, mj=0.7123, mjoc=2.13211),
        ),
        (
            OptimalChiPrentice14RootzoneStress,
            dict(tc=20, patm=101325, co2=400, vpd=1000, rootzonestress=0.5),
            dict(chi=0.62016),
        ),
        (
            OptimalChiC4,
            dict(tc=20, patm=101325, co2=400, vpd=1000),
            dict(chi=0.44967, mj=1.0, mjoc=1.0),
        ),
        (
            OptimalChiC4RootzoneStress,
            dict(tc=20, patm=101325, co2=400, vpd=1000, rootzonestress=0.5),
            dict(chi=0.37659, mj=1.0, mjoc=1.0),
        ),
        (
            OptimalChiLavergne20C3,
            dict(tc=20, patm=101325, co2=400, vpd=1000, theta=0.5),
            dict(beta=224.75255, chi=0.73663, mc=0.34911, mj=0.7258, mjoc=2.07901),
        ),
        (
            OptimalChiLavergne20C4,
            dict(tc=20, patm=101325, co2=400, vpd=1000, theta=0.5),
            dict(beta=24.97251, chi=0.44432, mc=0.28091, mj=1.0, mjoc=3.55989),
        ),
        (
            OptimalChiC4NoGamma,
            dict(tc=20, patm=101325, co2=400, vpd=1000),
            dict(chi=0.3919, mc=0.25626, mj=1.0),
        ),
        (
            OptimalChiC4NoGammaRootzoneStress,
            dict(tc=20, patm=101325, co2=400, vpd=1000, rootzonestress=0.5),
            dict(chi=0.31305, mc=0.21583, mj=1.0),
        ),
    ],
)
def test_subclasses(pmodelenv_args, subclass, expected):
    """Test that subclasses work as expected."""
    env = PModelEnvironment(**pmodelenv_args)
    instance = subclass(env)
    for key, value in expected.items():
        assert getattr(instance, key) == pytest.approx(value, rel=1e-3)


@pytest.mark.parametrize(
    argnames=["subclass", "extra_vars", "estimable_on_missing"],
    argvalues=[
        pytest.param(
            OptimalChiPrentice14,
            None,
            {
                "beta": ["tc", "vpd", "co2", "patm"],  # Fixed
                "xi": ["co2", "vpd"],  # Needs tc and patm
                "chi": [],  # Needs all
                "ci": [],
            },
            id="OptimalChiPrentice14",
        ),
        pytest.param(
            OptimalChiPrentice14RootzoneStress,
            {"rootzonestress": np.array([0.5])},
            {
                "beta": ["tc", "vpd", "co2", "patm", "rootzonestress"],  # Fixed
                "xi": ["co2", "vpd"],  # Needs tc and patm
                "chi": [],  # Needs all
                "ci": [],
            },
            id="OptimalChiPrentice14RootzoneStress",
        ),
        pytest.param(
            OptimalChiC4,
            None,
            {
                "beta": ["tc", "vpd", "co2", "patm"],  # Fixed
                "xi": ["co2", "vpd"],  # Needs tc and patm
                "chi": [],  # Needs all
                "ci": [],
            },
            id="OptimalChiC4",
        ),
        pytest.param(
            OptimalChiC4RootzoneStress,
            {"rootzonestress": np.array([0.5])},
            {
                "beta": ["tc", "vpd", "co2", "patm", "rootzonestress"],  # Fixed
                "xi": ["co2", "vpd"],  # Needs tc and patm
                "chi": [],  # Needs all
                "ci": [],
            },
            id="OptimalChiC4RootzoneStress",
        ),
        pytest.param(
            OptimalChiLavergne20C3,
            {"theta": np.array([0.5])},
            {
                "beta": ["tc", "vpd", "co2", "patm"],  # Needs theta
                "xi": ["co2", "vpd"],  # Needs tc and patm
                "chi": [],  # Needs all
                "ci": [],
            },
            id="OptimalChiLavergne20C3",
        ),
        pytest.param(
            OptimalChiLavergne20C4,
            {"theta": np.array([0.5])},
            {
                "beta": ["tc", "vpd", "co2", "patm"],  # Needs theta
                "xi": ["co2", "vpd"],  # Needs tc and patm
                "chi": ["co2"],  # chi does not depend on CO2
                "ci": [],  # But ci cannot be estimated without CO2
            },
            id="OptimalChiLavergne20C4",
        ),
        pytest.param(
            OptimalChiC4NoGamma,
            None,
            {
                "beta": ["tc", "vpd", "co2", "patm"],  # Needs theta
                "xi": ["co2", "vpd"],  # Needs tc and patm
                "chi": ["co2"],  # chi does not depend on CO2
                "ci": [],  # But ci cannot be estimated without CO2
            },
            id="OptimalChiC4NoGamma",
        ),
        pytest.param(
            OptimalChiC4NoGammaRootzoneStress,
            {"rootzonestress": np.array([0.5])},
            {
                "beta": ["tc", "vpd", "co2", "patm", "rootzonestress"],  # Needs theta
                "xi": ["co2", "vpd"],  # Needs tc and patm
                "chi": ["co2"],  # chi does not depend on CO2
                "ci": [],  # But ci cannot be estimated without CO2
            },
            id="OptimalChiC4NoGammaRootzoneStress",
        ),
    ],
)
def test_nan_handling(subclass, extra_vars, estimable_on_missing):
    """Test that subclasses handles NaNs correctly."""

    # Setup the required vars
    vars = dict(
        tc=np.array([20]),
        patm=np.array([101325]),
        co2=np.array([400]),
        vpd=np.array([1000]),
    )
    if extra_vars is not None:
        vars.update(extra_vars)

    # Set each var to nan in turn
    for var in vars:
        pmodelenv_args_copy = vars.copy()
        pmodelenv_args_copy[var] = np.array([np.nan])
        env = PModelEnvironment(**pmodelenv_args_copy)
        instance = subclass(env)

        for pred_var in ["beta", "xi", "chi", "ci"]:
            if var in estimable_on_missing[pred_var]:
                assert not np.isnan(
                    getattr(instance, pred_var)
                ), f"{pred_var} is np.nan but estimable with missing {var}"
            else:
                assert np.isnan(
                    getattr(instance, pred_var)
                ), f"{pred_var} estimated but should not be with missing {var}"
