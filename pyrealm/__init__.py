"""The pyrealm package.

TODO: complete this documentation
"""

import importlib.metadata
import os
import warnings

__version__ = importlib.metadata.version("pyrealm")


class ExperimentalFeatureWarning(Warning):
    """Warn about experimental features.

    This is just a simple wrapper on the base Warning to issue clearer warnings about
    experimental features in the code.
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return repr(self.message)


# Setup warnings to simpler one line warning
def warning_on_one_line(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """Provide a simple one line warning for use in docstrings."""
    filename = os.path.join("pyrealm", os.path.basename(filename))
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line
