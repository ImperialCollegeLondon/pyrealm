# Contributing to the `pyrealm` package

We welcome contributions to improving and extending the `pyrealm` package. This page
provides an overview of the key components for contributing to `pyrealm` and then links
to more details on the package website:

> [https://pyrealm.readthedocs.io/](https://pyrealm.readthedocs.io/)

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

> [!NOTE]
> See the [installation page](docs/source/development/installation.md) for more details
> on setting up the development environment and developing `pyrealm`.

## Development components

### The `pyrealm_build_data` package

The `pyrealm` repository also includes the `pyrealm_build_data` package providing data
files. This is only included in source distributions (`sdist`), so is not used by end
users, but provides data for testing the code, profiling performance and as inputs to
example usage in the package documentation.

> [!NOTE]
> See the [`pyrealm_build_data` page](docs/source/development/pyrealm_build_data.md) for
> more details.

### Code development

All code development is through this Github repository and the process is:

* Create an issue on Github describing the change to be made.
* Use the issue interface to generate a branch for the issue.
* Commit your changes on that branch.
* Create a pull request from that branch onto `develop`.
* Request a review from other package developers.
* Respond to reviews until the pull request is approved.
* Merge to `develop`.

The `pyrealm` package uses `mypy` to provide static type checking.

> [!NOTE]
> See the [code development page](docs/source/development/code_development.md) for
> more details.

### Code testing

We use the `pytest` package to provide testing for the `pyrealm` package. The test suite
includes unit tests, regression testing against alternate implementations and
performance profiling tests. We use `codecov` to assess the coverage of the package code
by the test suite.

> [!NOTE]
> See the [code testing page](docs/source/development/code_testing.md) for
> more details.

### Documentation

We use:

* The `sphinx` package to build the `pyrealm` website on ReadTheDocs.
* Docstrings to describe the code and automatically generate API descriptions for the
  website.
* The `doctest` package to provide limited examples of code use in docstrings.
* MyST Markdown notebooks to provide extended usage examples within the website.

> [!NOTE]
> See the [documentation page](docs/source/development/documentation.md) for
> more details.

### Code quality checks

The package includes configuration for a set of `pre-commit` hooks to ensure code
commits meet common community quality standards. The `pre-commit` tool blocks `git`
commits until all hooks pass. You should install these to ensure that your commited code
meets these standards.

> [!NOTE]
> See the [pre-commit configuration page](docs/source/development/pre_commit.md) for
> more details on `pre-commit` and the hooks used.

### Continuous integration

The project uses a GitHub Actions workflow to support continuous integration. The
workflow is used to check the code quality, apply code tests and performance checks and
build on all pull requests.

> [!NOTE]
> See the [continuous integration page](docs/source/development/pre_commit.md) for
> more details on the workflow and process.

### Release process

We use trusted publishing from Github releases to submit new versions of the package to
PyPI.

> [!NOTE]
> See the [release process page](docs/source/development/release_process.md) for
> more details on the release process.
