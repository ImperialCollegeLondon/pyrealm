# Contributing to the `pyrealm` package

## Quick Start

### Installation

The package requires Python 3.10 or newer. The `pyrealm` package can be installed
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
modules within `pyrealm` - see the [README](README.md#using-pyrealm) for links to these
examples - but typically a user will load input data, pass it into imported `pyrealm`
classes and functions and then use the outputs in further functions.

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

### Fetching the datasets using Git LFS

The datasets in `pyrealm_build_data` are stored via [Git LFS](https://git-lfs.com/).
After cloning the repository, only text pointers to these large files are downloaded.
In order to actually fetch the data into the local working tree,
you can run the command `git lfs pull`.

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
and check confirm that the package and website are building correctly. The
[`pyrealm_ci.yaml`](.github/workflows/pyrealm_ci.yaml) currently defines three CI jobs:

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
[`.pre-commit-config.yaml`](.pre-commit-config.yaml) shows the details of the tool
chain, but `isort`, `black`, `flake8` and `markdownlint` are used to maintain code
quality.

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

#### Using 'pytest-profiling'

[Pytest-profiling](https://pypi.org/project/pytest-profiling/) is a plugin to pytest
that enables profiling of tests. It can be used to generate a call graph and to
determine the number of hits and total time spent in each function or method.
[Graphviz](https://pypi.org/project/graphviz/) is a package that uses `dot` command to
facilitate the rendering of the call graph and a manual install of graphviz is required
depending on the os. Dedicated profiling tests have been created for PyRealm.
Please see the relevant testing directory.

```bash
poetry run pytest --profile
```

to generate a report. You can run

```bash
poetry run pytest --profile-svg
```

to generate a call graph. The graph is saved at `prof/combined.svg`.

To enable profiling of a test function or class, decorate it with `@pytest.mark.profiling`.

#### Using 'pytest-coverage'

Using the plugin [pytest-coverage](https://pypi.org/project/pytest-cov/) you can generate
coverage reports. You can run

```bash
poetry run pytest --cov=<test_path>
```

to perform coverage analysis. The report is stored with the name `index.html`.
It can be used to determine if your contribution is adequately tested.

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

Releasing a new version of the package uses trusted publishing. The process is described
on the package website in [this file](./docs/source/development/release_process.md).
