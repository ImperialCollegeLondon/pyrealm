---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  language: python
  name: pyrealm_python3
---

# Developing `pyrealm`

This page gives an overview of the process of contributing code to the `pyrealm`
package, along with the development environment and tools you will need to setup to work
with the codebase.

## Contributing code

The workflow for contributing code to `pyrealm` is:

1. Decide what you want to work on. This could be an existing bug or feature request or
   could be something new. If it is new, then create an new issue on Github describing
   what you want to change or add. The issue tracker provides templates for bugs and
   feature requests: please do provide as much detail as possible on the bug or the
   feature you would like to provide. If you want to work on an existing issue, then
   just add a comment and say you would like to work on it.

   Whatever issue you do want to work on, do give other developers a chance to comment
   on suggestions before putting a lot of effort in!

1. Use the Github issue page interface to create a branch for the issue. The branch name
   will start with the issue number, which makes branches much easier to track.

1. Check that branch out locally and make commits to it, pushing them to GitHub
   regularly. Do try and make frequent small commits with clear, specific commit
   messages.

1. Create a pull request (PR) from the issue branch onto the `develop` branch. The PR
   description should tag the issue being addressed and explain how the incoming code
   fixes the issue. You can start a PR as 'draft' PR: this can be a useful way to start
   describing a PR content and checking that testing is passing before opening a PR up
   for review.

1. Check that the continuous integration testing passes and fix any issues causing test
   failures.

1. Request reviews from other package developers. A PR cannot be merged into `develop`
   until at least one approving review has been added to the code. Reviews will often
   suggest changes to the code and you should discuss those suggestions and implement
   them.

1. Once a PR has been approved, the PR can be merged into `develop` and the branch can
   be deleted.

## The package development environment

The short descriptions below provide the key commands needed to set up your development
environment and provide links to more detailed descriptions of code development for
`pyrealm`.

### Python environment

You will need to install Python 3.10 or greater to develop `pyrealm`. We recommend using
`pyenv` or `pyenv-win` to manage your Python installations but this is only really
needed if you want to be able to test your code on different python versions.

### Package management

We use [`poetry`](https://python-poetry.org/docs/#installation) for dependency
management and for managing development environments and you will need to install it.

### Installing `pyrealm`

To develop `pyrealm`, you will also need to install [`git`](https://git-scm.com/) and
then clone the `pyrealm` GitHub repository.

```sh
git clone https://github.com/ImperialCollegeLondon/pyrealm.git
```

You can now use `poetry` to install the package dependencies. This is not just the
package requirements for end users of the package, but also a wider set of tools used in
package development. The `poetry` tools uses the
[pyproject.toml](https://github.com/ImperialCollegeLondon/pyrealm/blob/develop/pyproject.toml)
file to configure the dependencies that will be installed.

```bash
poetry install
```

Poetry uses a virtual environment for package development: all packages are installed to
a stand-alone python environment that is only used for `pyrealm` development. This makes
sure that the development environment is consistent across python versions and different
developers. It does mean you need to put `poetry run` before command to run commands in
that environment or `poetry shell` to start a new shell that specifically uses that
environment.

You should now be able to run the following command to see that `pyrealm` is installed
and is showing the current version.

```sh
poetry run python -c "import pyrealm; print(pyrealm.__version__)"
```

### Installing and using `pre-commit`

Development of the `pyrealm` package uses [`pre-commit`](https://pre-commit.com/). This
is a python tool that runs a set of checks on `git` commits and stops the commit from
completing when any of those checks fail. We use `pre-commit` to help catch a wide range
of common issues and make sure that all code pushed to the GitHub repository meets some
simple quality assurance checks and uses some common formatting standards.

The `pre-commit` tool is installed by the `poetry install` step above, so you now need
to install the `pyrealm` configuration for `precommit` and run the tool to set up the
environment and check it is all working.

```sh
poetry run pre-commit install
poetry run pre-commit run --all-files
```

That might take a little while to run on the first use. Once you have done this, every
`git commit` will generate similar output and your commit will fail if issues are found.
See the [code quality assurance page](./code_qa_and_typing.md) for more information on
the output, along with the configuration and update process.

### Static typing with `mypy`

The `python` programming language does not _require_ code objects to be typed, but the
`pyrealm` package uses [type hints](https://peps.python.org/pep-0484/) to annotate code.
Those type hints are then checked using the `mypy` static type checker, which is
installed by `poetry` and is run as one of the `pre-commit` checks.

The `mypy` package and the plugins we use are all installed by `poetry`. See the [code
quality assurance page](./code_qa_and_typing.md) for more information on using `mypy`.

### The `pyrealm-build-data` package

The `pyrealm` repository includes the [`pyrealm-build-data`
package](./pyrealm_build_data.md), which is used to provide a range of datasets used in
package testing and in documentation.

### Package testing

All code in the `pyrealm` package should have accompanying unit tests, using `pytest`.
Look at the existing test suite in the `tests/unit` directory to see the structure and
get a feel for what they should do, but essentially unit tests should provide a set of
known inputs to a function and check that the expected answer (which could be an
Exception) is generated.

Again, the `pytest` package and plugins are installed by `poetry`. See the [code testing
page](./code_testing.md) for more details.

### Code profiling

We use a specific set of `pytest` tests to carry out code profiling using
the `pytest-profile` plugin. This makes it easier to spot functions that are running
slowly within the `pyrealm` code. We also use our own `profiling/run-benchmarking.py`
script to compare code profiles between versions to check that new code has not impacted
performance.

See the [profiling and benchmarking page](./profiling_and_benchmarking.md) for more
details on running profiling.

### Documentation

We use `sphinx` to maintain the documentation for `pyrealm` and Google style docstrings
using the `napoleon` formatting to provide API documentation for the code. We use MyST
Markdown to provide dynamically built usage examples. See the [documentation
page](./documentation.md) for details.

In order to build the documentation, you will need to register the `poetry` virtual
enviroment, so that it can be used by `jupyter` and `myst` to build dynamic content.
This only needs to be done once.

```bash
poetry run python -m ipykernel install --user --name=pyrealm_python3
```

After that, the following code can be used to build the documentation

```bash
# Build docs using sphinx
cd docs
poetry run sphinx-build -W --keep-going source build
```

### GitHub Actions

We use GitHub Action workflows to update `pre-commit`, run code quality checks on pull
requests, and to automate profiling and release publication. See the [GitHub Actions
page](./github_actions.md) for details.

### Package version releases

We use trusted publishing from GitHub releases to release new versions of `pyrealm` to
[PyPI](https://pypi.org/project/pyrealm/). Releases are also picked up and archived on
[Zenodo](https://doi.org/10.5281/zenodo.8366847). See the [release process
page](./release_process.md) for details.
