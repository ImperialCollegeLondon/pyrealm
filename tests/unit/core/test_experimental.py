"""Tests the experimental decorator."""

import pytest


def test_experimental():
    """Test the experimental decorator."""
    from pyrealm.core.experimental import ExperimentalFeatureWarning2, experimental

    @experimental()
    def experimental_function():
        """The experimental_function."""
        return

    @experimental()
    class ExperimentalClass:
        """The ExperimentalClass."""

    class StandardClass:
        """The StandardClass."""

        @experimental()
        def experimental_method(self):
            """The experimental_method."""
            return

    with pytest.warns(ExperimentalFeatureWarning2):
        _ = experimental_function()
        assert ".. admonition:: Experimental" in experimental_function.__doc__

    with pytest.warns(ExperimentalFeatureWarning2):
        _ = ExperimentalClass()
        assert ".. admonition:: Experimental" in ExperimentalClass.__doc__

    inst = StandardClass()
    with pytest.warns(ExperimentalFeatureWarning2):
        inst.experimental_method()
        assert ".. admonition:: Experimental" in inst.experimental_method.__doc__
