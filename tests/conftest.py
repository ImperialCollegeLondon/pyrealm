"""Pytest configuration file."""

import pytest


def pytest_configure(config):
    """Add custom markers to pytest."""
    config.addinivalue_line(
        "markers", "profiling: mark test to run when profiling is enabled"
    )
    config.addinivalue_line(
        "markers", "profiling_only: mark test to run only when profiling is enabled"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on profiling markers."""
    if config.getoption("--profile"):
        skip_non_profile = pytest.mark.skip(reason="skip tests not for profiling")
        for item in items:
            if (
                "profiling_only" not in item.keywords
                and "profiling" not in item.keywords
                and item.name.startswith("test_")
            ):
                item.add_marker(skip_non_profile)
    else:
        skip_profile = pytest.mark.skip(reason="skip tests only for profiling")
        for item in items:
            if "profiling_only" in item.keywords:
                item.add_marker(skip_profile)
