"""Shared fixtures for demography testing."""

import pytest


@pytest.fixture
def fixture_flora():
    """Simple flora object for use in demography tests."""

    from pyrealm.demography.flora import Flora, PlantFunctionalType

    return Flora(
        [
            PlantFunctionalType(name="broadleaf", h_max=30),
            PlantFunctionalType(name="conifer", h_max=20),
        ]
    )
