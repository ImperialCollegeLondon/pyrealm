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

<!-- This file duplicates the content of CONTRIBUTING.md in the project root
but that document contains links that break by simply `include`ing it here -->

# Contributing to the `pyrealm` package

## Quick Start

### Installation

The package requires Python 3.9 or newer. The `pyrealm` package can be installed
from [PyPi](https://pypi.org/project/pyrealm/) but to develop the package you will need
to clone this repository and then install
[Poetry](https://python-poetry.org/docs/#installation) for dependency management and to
set up the development environment.

From within the cloned repository, first install the dependencies defined in
[pyproject.toml](https://github.com/ImperialCollegeLondon/pyrealm/blob/develop/pyproject.toml):

```bash
poetry install
```

Then you can run the tests to check that the installation is working correctly:

```bash
poetry run pytest
```

### Example package usage

The package is designed to provide a set of tools to be used within a Python script or
notebook. The package documentation provides some minimal examples of using the various
modules within `pyrealm` but typically a user will load input data, pass it into
imported `pyrealm` classes and functions and then use the outputs in further functions.

Some of the package modules - such as the `pmodel` and `splash` modules - require input
data in arrays, typically with spatial and temporal dimensions loaded from NetCDF files
or similar formats. Other modules - such as the `tmodel` - work on a site by site basis
and require the configuration of site-specific details, such as the definition of plant
functional types and plant community structures.

## The `pyrealm_build_data` package

The `pyrealm` repository includes both the `pyrealm` package and the
`pyrealm_build_data` package. The `pyrealm_build_data` package contains datasets that
are used in the `pyrealm` build process.

* Example datasets that are used in the package documentation, such as simple spatial
  datasets for showing the use of the P Model.
* "Golden" datasets including a set of input data and output predictions from
  previous implementations of `pyrealm` functionality. These are used to benchmark
  `pyrealm` functionality within the testing framework (see below).

Note that `pyrealm_build_data` is a source distribution only (`sdist`) component of
`pyrealm`, so is not included in binary distributions (`wheel`), such as those available
from PyPi.

## Code development

### Package management

The package uses `poetry` to manage dependencies, generate virtual environments for code
development and then for package building and publication. You will need to install
`poetry>=1.2.0` to develop the code.

### Source code management

The codebase is version-controlled using `git` with a repository at:

[https://github.com/ImperialCollegeLondon/pyrealm](https://github.com/ImperialCollegeLondon/pyrealm)

The package uses the `git flow` model for development and release:

* All code development should happen specific `feature/feature_name` branches.
* Pull requests (PR) should be made to merge feature branches into the `develop` branch.
* Candidate release versions should be made on specific `release/x.y.z` branches
  and these are then committed to the `main` branch only after final checking.
* The `main` branch should only ever contain commits representing new release
  versions - do not work on the `main` branch.

Both the `develop` and `main` branches are protected on GitHub to avoid accidents!

## Code documentation

### Docstrings

The module codes uses docstrings written in the [Google
style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
Unlike the main documentation pages, the docstrings in code are written using
reStructuredText because the `autodoc` functions in `sphinx` currently rely on `rst`
inputs. This allows the function documentation to be stored alongside the code and
included simply into the documentation.

### Package website and documentation

The `pyrealm` package is documented using `sphinx`, with source material in the
`docs/source` directory.

The documentation in `source` uses [Myst
Markdown](https://myst-parser.readthedocs.io/en/latest/) rather than the standard
`sphinx` reStructuredText (`.rst`) format. This is a literate programming format that
can be used as a Jupyter notebook. The markdown format is more `git` friendly than the
standard Jupyter `.ipynb` format, which uses a JSON format and is updated to contain
non-essential code execution information.

When the package documentation is built in `sphinx` the `myst_nb` extension is used
automatically to execute the code in MyST markdown documents and render the resulting
code and outputs. The documentation can therefore contain dynamically generated examples
of running code and output plots to demonstrate the use and behaviour of the package.
This build process also ensures that code in documentation is kept up to date with the
package. The syntax for including code blocks in documents is shown below - this example
includes a tag that causes the code to be concealed in the built documentation, but with
a toggle button to allow it to be expanded.

````{code-block}
```{code-cell} python
:tags: [hide-input]
# This is just an example code cell to demonstrate how code is included in 
# the pyrealm documentation.
```
````

## Development notes

The `sphinx` configuration includes the `sphinx.ext.mathjax` extension to support
mathematical notation. This has been configured to also load the `mhchem` extension,
supporting the rendering of chemical notation.

## Code quality and continuous integration

The project uses continuous integration (CI) via GitHub Actions to maintain code quality
and check confirm that the package and website are building correctly. The actions are
defined in `.github/workflows/pyrealm_ci.yaml` and currently include three CI jobs:

* code quality assurance (`qa`)
* code testing (`test`)
* documentation building (`docs_build`)

The status of code checking for pushed commits can be seen at:

[https://github.com/ImperialCollegeLondon/pyrealm/actions](https://github.com/ImperialCollegeLondon/pyrealm/actions)

Although GitHub Actions automates these steps for any pushes, pull requests and releases
on the repository, you should also perform the same steps locally before submitting code
to ensure that your code passes testing.

### Code quality assurance

The project uses `pre-commit` to enforce code quality. The configuration file
`.pre-commit-config.yaml` shows the details of the tool chain, but `isort`,`black`,
`flake8` and `markdownlint` are used to maintain code quality.

You will need to [install `pre-commit`](https://pre-commit.com/#install) to develop
package code. Once you have installed `pre-commit` you then need to install the
configuration for the repository:

```bash
pre-commit install
```

After this step, all commits within the repository must then pass the suite of quality
assurance checks. Note that the repository also uses a separate GitHub action to update
the code quality assurance toolchain on a weekly basis, so you will periodically see
updates to the checking.

### Code testing

The `pyrealm` package principally uses `pytest` to provide benchmark tests, unit tests
and integration testing. In addition, `doctest` is used to maintain examples of code
usage in the package docstrings and ensure that the documented return values are
correct.

#### Using `pytest`

The `test` directory contains modules providing test suites for each of the different
package modules. This includes:

* benchmark testing the output of `pyrealm` code against previously existing
  implementations of some functionality, such as the `rpmodel` and `SPLASH` packages.
* unit testing of individual functions and methodsm and,
* integration testing of using combinations of modules.

These are the main tests that the package is behaving as expected and producing stable
outputs and can be run from repository using:

```bash
poetry run pytest
```

#### Using `doctest`

The package docstrings contain `doctest` examples of code use. These examples are
intended to provide simple examples of method or function use and generate an output and
the `doctest` process is used to validate that the provided values are correct.

These tests are automatically run when `pytest` is run, but individual module files can
be checked using, for example:

```bash
poetry run python -m doctest pyrealm/pmodel/pmodel.py
```

Normally, `doctest` is used to test a return value but can also be used to check that an
error or warning is raised. See the example for `pyrealm.utilities.convert_rh_to_vpd`
which redirects the stderr to stdout to allow for the warning text to be included in the
doctest.

### Code profiling

`pytest-profiling` is used for profiling the code with large datasets. Only functions or
classes decorated by `@pytest.mark.profiling` will run during profiling.

When a commit of the `develop` branch is pushed or released, the CI workflow will 
automatically run code profiling and save the profiling results in the `profiling/` folder.

To run profiling manually, use

```bash
poetry run pytest --profile/--profile-svg
```

The results will be saved at `prof/`. However, the generate `.prof` files are not human
readable. In order to append the new profiling results to the CSV report at 
`profiling/prof-report.csv`, and generate some benchmark plots, run

```bash
poetry run python profiling/report.py
```

#### Latest profiling results

![call graph](/profiling/call-graph.svg)
![profiling plot](/profiling/profiling.png)
![benchmark plot](/profiling/benchmark.png)

### Building the documentation

`sphinx` is used to build an HTML version of the package documentation provided in
`docs/source` and to build the API documentation provided in the code docstrings. The
`sphinx` building process requires some extra packages, but these are included in the
`docs` group in `pyproject.toml` and should be installed.

In order to build the package documentation, Jupyter needs to be able to associate the
documentation files with the Python environment managed by `poetry`. This is done by
installing the `poetry` environment as a new Jupyter kernel with a fixed name. This
allows all build systems to run notebooks using the correct build environment:

```bash
# Set ipython kernel
poetry run python -m ipykernel install --user --name=pyrealm_python3
```

In order to build the package documentation, the following command can be used:

```bash
# Build docs using sphinx
cd docs
poetry run sphinx-build -W --keep-going source build
```

### Online documentation

The documentation for the package is hosted at:

[https://pyrealm.readthedocs.io/en/develop/pmodel.html](https://pyrealm.readthedocs.io/en/develop/pmodel.html)

This has been configured to automatically build commits to the `main` branch, which
should generate version specific sets of documentation.

### Referencing

The documentation uses the `sphinxcontrib-bibtex` package to support citations. This
uses Latex like citation keys in the documentation to insert references and build a
bibliography. The `sphinx` configuration in `docs/source/conf.py` provides a custom
Author/Year citation style. The reference library in `source/refs.bib` needs to be kept
up to date with the literature for the project.

## Release process

Releasing a new version of the package follows the flow below:

1. Create a `release` branch from `develop` containing the new release code.
2. Check that this branch builds correctly, that the documentation builds correctly and
   that the package publishes to the `test-pypi` repository.
3. When all is well, create pull requests on GitHub to merge the `release` branch into
   both `develop` and `main`, along with a version tag for the release.
4. Once you have updated your local repository, then the tag can be used to build and
   publish the final version to the main PyPi site.

It is easier if `git` is configured to push new tags along with commits. This
essentially just means that new releases can be sent with a single commit. This only
needs to be set once.

```bash
set git config --global push.followTags true
```

### Create the release branch

Using `git flow` commands as an example to create a new release:

```sh
git flow release start new_release
```

Obviously, use something specific, not `new_release`! Ideally, you would do a dry run of
the next step and use the version - but it should be fairly obvious what this will be!

The `poetry version` command can then be used to bump the version number. Note that the
command needs a 'bump rule', which sets which part of the semantic version number to
increment (`major`, `minor` or `patch`). For example:

```sh
poetry version patch
```

This updates `pyproject.toml`. At present, the package is set up so that you also *have
to update the version number in `pyrealm/version.py`* to match manually.

### Publish and test the release branch

With those changes committed, publish the release branch:

```sh
git commit -m "Version bump" pyrealm/version.py
git flow release publish new_release
```

The GitHub Actions will then ensure that the code passes quality assurance and then runs
the test suites on a range of Python versions and operating systems.

### Check package publication

The `sdist` and `wheel` builds for the package can then be built locally using `poetry`

```bash
poetry build
```

The first time this is run, `poetry` needs to be configured to add the Test PyPi
repository and an API token from that site. Note that accounts are not shared between
the Test and main PyPi sites: the API token for `test-pypi` is different from
`pypi` and you have to log in to each system separately and generate a token on each.

```sh
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry config pypi-token.test-pypi <your-token>
```

The built packages can then be published to `test-pypi` using:

```sh
poetry publish -r test-pypi
```

### Check the documentation builds

Log in to:

[https://readthedocs.org](https://readthedocs.org)

which is the admin site controlling the build process. From the Versions tab, activate
the `release/new_release` branch and wait for it to build. Check the Builds tab to see
that it has built successfully and maybe check updates! If it has built succesfully, do
check pages to make sure that page code has executed successfully, and then go back to
the Versions tab and deactivate and hide the branch.

### Create pull requests into `main` and `develop`

If all is well, then two PRs need to be made on GitHub:

* The `release` branch into `main`, to bring all commits since the last release and
  any fixes on release into `main`.
* The `release` branch into `develop`, to bring any `release` fixes back into `develop`.

Once both of those have been merged, the `feature` branch can be deleted.

### Tag, build and publish

Once the `origin` repository is merged, then use `git pull` to bring `develop` and
`main` up to date on a local repo. Then, create a tag using the release version.

```sh
git checkout main
git tag <version>
git push --tags
```

The final commit on `main` is now tagged with the release version. You can add tags on
the GitHub website, but only by using the GitHub release system and we are using PyPi to
distribute package releases.

Before publishing a package to the main PyPi site for the first time, you need to set an
API token for PyPi.

```sh
poetry config pypi-token.pypi <your-token>
```

And now you can build the packages from `main` and publish.

```sh
poetry build
poetry publish
```
