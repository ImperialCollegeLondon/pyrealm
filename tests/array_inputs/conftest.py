"""Allows setting of the DEBUG flag using a command line argument."""


def pytest_configure(config):
    """Updates the DEBUG flag if --debug-tests is used."""
    import tests.array_inputs.utils

    debug = config.getoption("--debug-tests", default=False)
    tests.array_inputs.utils.config.DEBUG = debug

    if debug:
        config.option.capture = "no"  # equivalent to -s
        config.option.tbstyle = "short"  # equivalent to --tb=short
