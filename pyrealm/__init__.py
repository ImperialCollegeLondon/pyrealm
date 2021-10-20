import os
import sys
import warnings

# _ROOT = os.path.abspath(os.path.dirname(__file__))
#
#
# def get_data_path(path):
#     return os.path.join(_ROOT, 'data', path)

# Setup warnings to simpler one line warning

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    
    filename = os.path.join('pyrealm', os.path.basename(filename))
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)

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