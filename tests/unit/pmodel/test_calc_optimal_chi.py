"""Testing CalcOptimal submodule."""

import numpy as np
import pytest

from pyrealm.constants import PModelConst
from pyrealm.pmodel import CalcOptimalChi, PModelEnvironment


@pytest.fixture
def rootzonestress():
    """Return the root zone stress."""
    return np.array([1])
    # return np.array([0])


@pytest.fixture
def const():
    """Return the PModelConst."""
    return PModelConst()


@pytest.mark.parametrize(
    argnames=["pmodelenv_args", "pmodel_args", "expected"],
    argvalues=[
        (  # Single site, C3 prentice14
            dict(tc=20, patm=101325, co2=400, vpd=1000),
            dict(method_optchi="prentice14"),
            dict(
                # beta=24.97251,
                chi=0.69435,
                mc=0.33408,
                mj=0.7123,
                mjoc=2.13211,
            ),
        ),
        (  # Single site, C3 lavergne20
            dict(tc=20, patm=101325, co2=400, vpd=1000, theta=0.5),
            dict(method_optchi="lavergne20_c3"),
            dict(
                beta=224.75255,
                chi=0.73663,
                mc=0.34911,
                mj=0.7258,
                mjoc=2.07901,
            ),
        ),
        (  # Single site, C4
            dict(tc=20, patm=101325, co2=400, vpd=1000),
            dict(method_optchi="c4"),
            dict(
                # beta=21.481,
                chi=0.44967,
                mc=1.0,
                mj=1.0,
                # mjoc=-19.619,
            ),
        ),
        # (  # Single site, C4 no gamma
        #     dict(tc=20, patm=101325, co2=400, vpd=1000),
        #     dict(method_optchi="c4_no_gamma"),
        #     dict(
        #         # beta=21.481,
        #         chi=0.3919,
        #         mc=0.3,
        #         mj=1.0,
        #         # mjoc=-19.619,
        #     ),
        # ),
        (  # Single site, C4 lavergne20
            dict(tc=20, patm=101325, co2=400, vpd=1000, theta=0.5),
            dict(method_optchi="lavergne20_c4"),
            dict(
                beta=24.97251,
                chi=0.44432,
                mc=0.28091,
                mj=1,
                mjoc=3.55989,
            ),
        ),
        # (  # Single site, C3 prentice14 with all-zero inputs
        #     dict(tc=20, patm=101325, co2=400, vpd=1000),
        #     dict(method_optchi="prentice14"),
        #     dict(
        #         # beta=24.97251,
        #         chi=0,
        #         mc=0,
        #         mj=0,
        #         mjoc=0,
        #     ),
        # ),
        # (  # Single site, C3 prentice14 with all-zero inputs
        #     dict(tc=20, patm=101325, co2=400, vpd=1000),
        #     dict(method_optchi="prentice14"),
        #     dict(
        #         # beta=24.97251,
        #         chi=np.nan,
        #         mc=np.nan,
        #         mj=np.nan,
        #         mjoc=np.nan,
        #     ),
        # ),
    ],
)
def test_calcoptimalchi(pmodelenv_args, pmodel_args, expected, rootzonestress, const):
    """Test CalcOptimalChi."""
    env = PModelEnvironment(**pmodelenv_args)
    calc_optimal_chi = CalcOptimalChi(
        env, rootzonestress, pmodel_args["method_optchi"], const
    )

    for key in expected:
        assert np.allclose(getattr(calc_optimal_chi, key), expected[key], atol=0.0001)
