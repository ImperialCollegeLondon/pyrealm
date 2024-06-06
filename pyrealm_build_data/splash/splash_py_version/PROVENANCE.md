# Provenance of the splash_py_version directory

This directory contains the original Python implementation of the SPLASH model,
developed by Tyler Davis, hosted at [https://bitbucket.org/labprentice/splash]. The
files here are the contents of the path: `releases/v1.0/py_version` from the `master`
branch of the repo at commit `52d9454b566d`.

Changes:

* Two files have been added to the directory. The first is this provenance note and the
  other is the `__init__.py` file to allow the module to be imported by the benchmarking
  scripts.
* The code files have been modified by `isort` and `black` to make it easier to get the
  files past `pre-commit` but the `flake8` and `mypy` checking are explicitly
  suppressed to keep the files as close to their original content as possible.
* The package internal imports have been prepended with `splash_py_version` to allow the
  code to be imported as a package by the benchmark data generation scripts.
* The data path in `main.py` has been updated to point to the location of the input in
  this repo, rather than the original repo.
