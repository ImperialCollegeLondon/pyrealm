"""Provides some simple initialisation tests for the TwoLeaf classes."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="args,outcome,msg",
    argvalues=(
        pytest.param(
            dict(
                solar_elevation=np.array([0.6]),
                ppfd=np.array([1000]),
                leaf_area_index=np.array([2.0]),
                patm=np.array([101325]),
            ),
            does_not_raise(),
            None,
            id="all good single site",
        ),
        pytest.param(
            dict(
                solar_elevation=np.array([0.6, 0.7]),
                ppfd=np.array([1000, 1600]),
                leaf_area_index=np.array([2.0, 5.0]),
                patm=np.array([101325, 95608]),
            ),
            does_not_raise(),
            None,
            id="all good two sites",
        ),
        pytest.param(
            dict(
                solar_elevation=np.array([0.6, 0.7]),
                ppfd=np.array([1000, 1600]),
                leaf_area_index=np.array([2.0, 5.0]),
                patm=np.array([101.325, 95.608]),
            ),
            pytest.warns(UserWarning, match=r"values outside the expected"),
            None,
            id="out of bounds",
        ),
        pytest.param(
            dict(
                solar_elevation=np.array([0.6, 0.7]),
                ppfd=np.array([1000, 1600, 4567]),
                leaf_area_index=np.array([2.0, 5.0]),
                patm=np.array([101.325, 95.608]),
            ),
            pytest.raises(ValueError),
            "Inputs contain arrays of different shapes.",
            id="shape mismatch",
        ),
    ),
)
def test_TwoLeafIrradiance(args, outcome, msg):
    """Tests initialisation conditions of TwoLeafIrradiance."""
    from pyrealm.pmodel.two_leaf import TwoLeafIrradiance

    with outcome as outc:
        two_leaf = TwoLeafIrradiance(**args)

        # Check the calculated attributes have been populated and have the right shape
        assert hasattr(two_leaf, "sunlit_absorbed_irradiance")
        assert two_leaf.sunlit_absorbed_irradiance.shape == args["ppfd"].shape

        return

    # Handle exception messages
    assert str(outc.value) == msg


def test_TwoLeafAssimilation():
    """Test the creation of a TwoLeafAssimilation instance."""

    from pyrealm.pmodel import PModel, PModelEnvironment
    from pyrealm.pmodel.two_leaf import TwoLeafAssimilation, TwoLeafIrradiance

    model = PModel(
        env=PModelEnvironment(
            tc=np.array([20]),
            vpd=np.array([800]),
            patm=np.array([100000]),
            co2=np.array([400]),
            fapar=np.array([1]),
            ppfd=np.array([1500]),
        )
    )

    irrad = TwoLeafIrradiance(
        solar_elevation=np.array([0.7]),
        patm=np.array([100000]),
        ppfd=np.array([1500]),
        leaf_area_index=np.array([3]),
    )

    _ = TwoLeafAssimilation(pmodel=model, irradiance=irrad)
