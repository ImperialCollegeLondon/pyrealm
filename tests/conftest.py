"""Global configuration options for pytest."""


def pytest_addoption(parser):
    """Allow profiling dataset sizes to be scaled from the command line."""
    parser.addoption("--pmodel-profile-scaleup", type=int, default=6)
    parser.addoption("--splash-profile-scaleup", type=int, default=125)

    # Flag for debugging array_inputs tests
    parser.addoption("--debug-tests", action="store_true", default=False)
