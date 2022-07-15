"""The pyrealm package.

TODO: complete this documentation
"""

import os
import warnings

# _ROOT = os.path.abspath(os.path.dirname(__file__))
#
#
# def get_data_path(path):
#     return os.path.join(_ROOT, 'data', path)


class ExperimentalFeatureWarning(Warning):
    """Warn about experimental features.

    This is just a simple wrapper on the base Warning to issue clearer warnings about
    experimental features in the code.
    """

    def __init__(self, message) -> None:
        self.message = message

    def __str__(self) -> str:
        return repr(self.message)


# Setup warnings to simpler one line warning
def warning_on_one_line(
    message: str, category: str, filename: str, lineno: int, file=None, line=None
):
    """Provide a simple one line warning for use in docstrings."""
    filename = os.path.join("pyrealm", os.path.basename(filename))
    return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line

# # And provide a decorator to catch warnings in doctests
# # https://stackoverflow.com/questions/2418570/

# def stderr_to_stdout(func):
#     def wrapper(*args):
#         stderr_bak = sys.stderr
#         sys.stderr = sys.stdout
#         try:
#             return func(*args)
#         finally:
#             sys.stderr = stderr_bak
#     return wrapper
