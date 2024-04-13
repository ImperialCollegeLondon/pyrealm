# Contributing to the `pyrealm` package

We welcome contributions to improving and extending the `pyrealm` package. This page
provides an overview of the key components for contributing to `pyrealm` and then links
to more details on the package website: [https://pyrealm.readthedocs.io/](https://pyrealm.readthedocs.io/)

## Quick Start

The `pyrealm` package uses the `poetry` tool for package management. Getting a simple
development environment should involve  installing Python 3.10 or higher and `poetry`
and then using the following commands:

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

See the [installation page](docs/source/development/installation.md) for more details
on setting up the development environment and developing `pyrealm`.

## Development components

### Contributing code

All code contributions are added to `pyrealm` through this Github repository. You might
want to help fix an existing bug or add a requested feature or have your own ideas to
add to `pyrealm`. Whatever you want to do the first, thing is to open a new issue or
post on an existing issue and start a conversation with the other developers. We are
really keen to engage with new contributors and to help get new code into `pyrealm` in
the smoothest possible way.

See the [code development page](docs/source/development/code_development.md) for more
details.

### Code testing

We use the `pytest` package to provide testing for the `pyrealm` package. The test suite
includes unit tests, regression testing against alternate implementations and
performance profiling tests. We use `codecov` to assess the coverage of the package code
by the test suite.

See the [code testing page](docs/source/development/code_testing.md) for more details.

### Documentation

We use:

* The `sphinx` package to build the `pyrealm` website on ReadTheDocs.
* Docstrings to describe the code and automatically generate API descriptions for the
  website.
* The `doctest` package to provide limited examples of code use in docstrings.
* MyST Markdown notebooks to provide extended usage examples within the website.

See the [documentation page](docs/source/development/documentation.md) for more details.

### The `pyrealm_build_data` package

The `pyrealm` repository also includes the `pyrealm_build_data` package that is used to
provide data files for use in testing and documentation.

See the [`pyrealm_build_data` page](docs/source/development/pyrealm_build_data.md) for
more details on the available data and usage.

### Code quality checks

The package includes configuration for a set of `pre-commit` hooks to ensure code
commits meet common community quality standards. The `pre-commit` tool blocks `git`
commits until all hooks pass. You should install these to ensure that your commited code
meets these standards.

See the [pre-commit configuration page](docs/source/development/pre_commit.md) for more
details on setting up `pre-commit` and the hooks used.

### Continuous integration

The project uses a GitHub Actions workflow to support continuous integration. The
workflow is used to check the code quality, apply code tests and performance checks and
build on all pull requests.

See the [continuous integration page](docs/source/development/pre_commit.md) for
more details on the workflow and process.

### Release process

We use trusted publishing from Github releases to submit new versions of the package to
PyPI.

See the [release process page](docs/source/development/release_process.md) for
more details on the release process.

### Profiling and benchmarking

Even if the code works as expected and passes all our tests, it can still be slow! We
use code profiling to work out where time is spent when using `pyrealm` and identify
where we can improve performance. We also use benchmarking between `pyrealm` versions to
make sure that changes to the code aren't making it slower. This is run automatically on
the `develop` branch but can also be used to do local profiling and benchmarking.

See the [profiling and benchmarking
page](docs/source/development/profiling_and_benchmarking.md) for more details.
