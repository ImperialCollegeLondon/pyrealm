---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

# Package testing

The `pyrealm` package uses `pytest` to provide benchmark tests, unit tests and
integration testing. In addition, `doctest` is used to maintain examples of code usage
in the package docstrings and ensure that the documented return values are correct.

## Using `pytest`

The `tests` directory contains modules providing test suites for each of the different
package modules. This includes:

* Regression testing the output of `pyrealm` code against previously existing
  implementations of some functionality, such as the `rpmodel` and `SPLASH` packages.
* Unit testing of individual functions and methods.
* Property-based testing for functions that take array inputs to check broadcasting and
  conversion of xarray inputs. See the [section
  below](#resolving-errors-in-the-array-inputs-tests) for tips on resolving errors in
  these tests.
* [Profiling](./profiling_and_benchmarking.md) and integration testing using
  combinations of modules.

These are the main tests that ensure that the package is behaving as expected and that
it produces stable outputs. The test suite can be run from repository using:

```bash
poetry run pytest
```

The `setup.cfg` file contains `pytest` configuration details. By default, we do not
include the slow-running `profiling` tests as part of the standard `pytest` suite. See
the [profiling page](./profiling_and_benchmarking.md) for details on running those
tests.

## Using `doctest`

Some of the package docstrings contain `doctest` examples of code use. These examples
are intended to provide simple examples of method or function use and generate an
output: the `doctest` module is used to make sure that the code runs and gives the
expected result.

We have configured `pytest` to automatically also run `doctest`, but you can manually
check the tests in files using, for example:

```bash
poetry run python -m doctest pyrealm/pmodel/pmodel.py
```

Normally, `doctest` is just used to test a return value: the value tested is the value
printed to the console, so it is common to use some form of `round` to make sure values
match. It can also be used to check that an error or warning is raised. See the
docstring for {meth}`~pyrealm.core.hygro.convert_rh_to_vpd` to see how checking for
warning text can be included in a doctest.

## Using `pytest-coverage` and `codecov`

Using the plugin [pytest-coverage](https://pypi.org/project/pytest-cov/) you can
generate coverage reports. You can run:

```bash
poetry run pytest --cov=<test_path>
```

to perform coverage analysis. The report is stored with the name `index.html`. It can be
used to determine if your contribution is adequately tested. The GitHub Actions
[continuous integration workflow](./github_actions.md#pyrealm_ciyaml) automatically
uploads coverage data to the
[CodeCov](https://app.codecov.io/gh/ImperialCollegeLondon/pyrealm) website.

(array-inputs-errors)=

## Resolving errors in the array inputs tests

Any functions that take `numpy.array` arguments will automatically be tested for correct
broadcasting of [valid input shapes](../users/array_inputs.md#array-shapes), and
functions that take `ArrayType` arguments will test for correct conversion of
`xarray.DataArray` inputs.

The `tests/array_inputs/overrides.py` file can be used to resolve errors in these tests,
using:

* `SKIP_METHODS` - a list of functions / methods to skip because they are not relevant
  or have issues that are difficult to resolve.

* `IGNORE_OUTPUTS` - a list of function results or class attributes to skip when
  checking for equality as they are not expected to be equal. This is only used by the
  broadcasting tests.

* `ADDITIONAL_INIT_METHODS` - a dictionary containing any additional methods that need
  to be used when initialising objects of a certain class.

* `REQUIRES` - a dictionary to define keyword arguments that must be used in each
  function / method. This can also be used to avoid using default values for non-keyword
  arguments.

* `MANUAL_ARGS` - a dictionary containing manually defined arguments for functions /
  methods where the automatic generation doesn't work.

The latter is used to resolve most errors. Such as where parameters require specific
values (e.g. if a `float` parameter fails for the default value of 1), or, most
commonly, if multiple array arguments must share the same shape or size of a dimension.
For example, the code below shows using `register_args` to update `MANUAL_ARGS` for the
case where two array inputs (`a` and `b`) to `Class.method` need the same time dimension
and are not allowed to be broadcastable (length 1):

```python
@register_args("Class.method")
def _(ctx):
    shape = _set_time_len(3, ctx, allow_one=False)
    return {"a": np.ones(shape), "b": np.ones(shape)}
```

It is worth noting that in this example `a` and `b` do not necessarily have the same
shape (aside from the time dimension) because the function will be called separately
when instantiating each argument with a different shape each time.

### Debugging

When errors have occurred they can be investigated using the `--debug-tests` flag. This
can be used together with the `-k [test_name]` option to debug a single failing test,
or, to rerun all tests that failed using debug mode:

```bash
pytest tests/array_inputs --lf --debug-tests
```

This will print the arguments generated for each function and whether they are default
arguments (default), manually defined by `MANUAL_ARGS` (manual), or automatically
generated (automatic). The data of arrays will not be output, but their shape and
dimension names (for xarray) can be used to check for mismatched dimensions.

Another useful tool is [pytest-pudb](https://pypi.org/project/pytest-pudb/) which adds
the `--pudb` flag. This will drop you into an interactive debugging session when the
tests fail. This can be used to inspect the state and what caused the failure. This may
break for recent versions of python, but [pdb can still be
used](https://docs.pytest.org/en/stable/how-to/failures.html#pdb-option).

If an error is difficult to resolve then add the function name to the `SKIP_METHODS`
list. Although, this should ideally not be used as a first port of call.
