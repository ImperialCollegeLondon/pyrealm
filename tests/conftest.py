"""Global configuration options for pytest."""


def pytest_addoption(parser):
    """Allow profiling dataset sizes to be scaled from the command line."""
    parser.addoption("--pmodel-profile-scaleup", type=int, default=8)
    parser.addoption("--splash-profile-scaleup", type=int, default=200)
