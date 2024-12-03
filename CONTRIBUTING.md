# Contributing to the `pyrealm` package

We welcome contributions to improving and extending the `pyrealm` package. This page
provides an overview of the key components for contributing to `pyrealm` and then links
to more details on the package website:
[https://pyrealm.readthedocs.io/](https://pyrealm.readthedocs.io/)

The [development overview
page](https://pyrealm.readthedocs.io/en/latest/development/overview.html) on the website
provides an expanded overview of this page, including more details on setting up your
computer to develop `pyrealm`.

## Contributing code

All code contributions are added to `pyrealm` through this Github repository. You might
want to help fix an existing bug or add a requested feature or have your own ideas to
add to `pyrealm`. Whatever you want to do, the first thing is to open a new issue or
post on an existing issue and start a conversation with the other developers. We are
really keen to engage with new contributors and to help get new code into `pyrealm` in
the smoothest possible way.

See the [development overview
page](https://pyrealm.readthedocs.io/en/latest/development/overview.html#contributing-code)
for more details.

## Quick Start

The `pyrealm` package uses the `poetry` tool for package management. Getting a simple
development environment should involve:

* Installing `python`: `pyrealm` requires `python >3.10` and we currently support
  versions `python <=3.11`.
* Installing `poetry`: although the `poetry` tool manages the versions of all the
  package requirements, the version of `poetry` itself is not managed. Developers should
  all ensure that they install the **same version of poetry** to avoid version
  incompatibility issues. We currently use `poetry 1.8.2`.

```sh
# Download the repository
git clone https://github.com/ImperialCollegeLondon/pyrealm.git
# Move into the repository root
cd pyrealm
# Use poetry to install the package requirements
poetry install
# Check the installation has succeeded.
poetry run pytest
```

See the [installation
page](https://pyrealm.readthedocs.io/en/latest/development/overview.html) for more
details on setting up the development environment and developing `pyrealm`.

### The `pyrealm_build_data` package

The `pyrealm` repository also includes the `pyrealm_build_data` package that is used to
provide data files for use in testing and documentation. See the [`pyrealm_build_data`
page](docs/source/development/pyrealm_build_data.md) for more details on the available
data and usage.

### Code testing

We use the `pytest` package to provide unit testing for the `pyrealm` package. The test
suite lives in the `tests` directory and all new and updated code should include tests
to check that the code works as intended. The test suite also includes regression
testing against previous implementations and a set of tests for code profiling and
performance benchmarking.

We use the `pytest-cov` plugin to generate coverage reports on `pyrealm` code, to check
how well the code is covered by the test suite. We use `codecov` to record code coverage
and report on changes in coverage between commits..

See the [code testing
page](https://pyrealm.readthedocs.io/en/latest/development/code_testing.md) for more
details.

### Code quality checks

The package includes configuration for a set of `pre-commit` hooks to ensure code
commits meet common community quality standards. The `pre-commit` tool blocks `git`
commits until all hooks pass. You should install these to ensure that your commited code
meets these standards.

See the [pre-commit configuration
page](https://pyrealm.readthedocs.io/en/latest/development/code_qa_and_typing.md) for
more details on setting up `pre-commit` and the hooks used.

### Continuous integration

The project uses a GitHub Actions workflow to support continuous integration. The
workflow is used to check the code quality, run the unit and regression testing suite
and check the documentation builds successfully on all pull requests.

See the [GitHub Actions
page](https://pyrealm.readthedocs.io/en/latest/development/github_actions.md) for more
details on the workflow and process.

### Profiling and benchmarking

Even if the code works as expected and passes all our tests, it can still be slow! We
use code profiling to work out where time is spent when using `pyrealm` and identify
where we can improve performance. We also use benchmarking between `pyrealm` versions to
make sure that changes to the code aren't making it slower.

This is currently run manually using the `performance_regression_checking.sh` script in
the `profiling` directory. When this bash script is run without an argument, the current
`HEAD` will be compared to the `origin/develop` branch. Alternatively, the two commits
to be compared can be provided as parameters with `-n` for the "new" commit (the current
HEAD, or code you have changed), and `-o` for the "old" commit (the baseline code you
want to compare the perfomance against). The code will fail with an error message if the
new performance is more than 5% slower than the baseline, otherwise it will succeed with
a message indicating whether the new code is faster or has similar performance.

```sh
cd profiling
poetry run performance_regression_checking.sh
```

To look deeper into how much time is spent in each function, the command

```bash
poetry run pytest --profile-svg -m "profiling"
```

will run the profiling tests and generate a call graph and table with this information in
the `profiling` subdirectory.

### Documentation

We use:

* The `sphinx` package to build the `pyrealm` website on ReadTheDocs.
* Docstrings to describe the code and automatically generate API descriptions for the
  website.
* The `doctest` package to provide limited examples of code use in docstrings.
* MyST Markdown notebooks to provide extended usage examples within the website.

See the [documentation
page](https://pyrealm.readthedocs.io/en/latest/development/documentation.md) for more
details.

### Release process

We use trusted publishing from Github releases to submit new versions of the package to
PyPI. See the [release process
page](https://pyrealm.readthedocs.io/en/latest/development/release_process.md) for more
details on the release process.
