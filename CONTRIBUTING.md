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

### Extensions

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
* performance profiling of the `pyrealm` codebase: see the separate [section on
  profiling](./CONTRIBUTING#profiling).

These are the main tests that the package is behaving as expected and producing stable
outputs and can be run from repository using:

```bash
poetry run pytest
```

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

### Profiling and benchmarking

We use code profiling to assess the performance of `pyrealm` code and compare it to
previous code versions to identify bottlenecks and guard against degraded performance
when code changes.

Profiling and benchmarking can be run manually - which is useful if you are working on
code and want to check it doesn't impact performance - but are also run as part of the
continuous integration process when code is to be merged into the `develop` or `main`
branches.

#### Generating profiling data

We use the [pytest-profiling](https://pypi.org/project/pytest-profiling/) plugin to run
a set of profiling tests and generate profiling data. These tests are located
`tests/profiling` and consist of a small set of high-level scripts that are intended to
use a large proportion of the `pyrealm` codebase with reasonably large inputs.

All tests in the profiling suite are decorated with `@pytest.mark.profiling`. Tests with
this mark are excluded from the standard `pytest` testing via the `-m "not profiling"`
argument in `setup.cfg`. Any test can be decorated with the `profiling` mark to move it
temporarily into the profiling test suite.

To run the profiling test suite and generate code profiling data, run `pytest` as
follows:

```bash
poetry run pytest --profile-svg -m "profiling"
```

This selects _only_ the profiling tests and runs them using `pytest-profiling`. The
`--profile-svg` both runs the profiling _and_ generates a figure showing the call
hierachy of code objects and the time time spent in each call. Generating this graph
graph requires the [graphviz](https://pypi.org/project/graphviz/) command line library,
which provides the `dot` command for generating SVG graph diagrams. You will need to
install `graphviz` to use this option. Alternatively, you can use the following command
to only generate the profile data.

```bash
poetry run pytest --profile -m "profiling"
```

The `pytest-profiling` plugin saves data and graphs to the `prof` direectory, which is
excluded from the `git` repository. The key files are the combined results:
`prof/combined.prof` and `prof/combined.svg`.

#### Benchmarking code performance

When `pytest-profiling` runs, the resulting `prof/combined.prof` file contains detailed
information on all the calls invoked in the test code, including the number of times
each call is made and the time spent on each call. The `prof/combined.svg` shows where
time is spent during the test runs, which identifies bottlenecks, but it is also useful
to check that the time spent on a call has not increased markedly when code is revised.

The `profiling` directory contains a database of previous profiling results
(`profiling-database.csv`) and the `run_benchmarking.py` tool, which can be used to
benchmark new profiling data. The basic process is:

* read the current `prof/combined.prof` file and convert it to human-readable CSV data,
* find the best performance for each call over recent previous runs, and
* check if the relative performance of the incoming code is notably slower than that
  best performance.

By default, we use the 5 most recent code versions and the threshold performance is 5%
slower than the previous best performance. The report tool consists of a command line
wrapper around a set of profiling functions, which can be imported for programatic use.
for use.

The usage of the tool is:

```txt
usage: run_benchmarking.py [-h] [--exclude EXCLUDE] [--n-runs N_RUNS]
                           [--tolerance TOLERANCE] [--append-on-pass] [--new-database]
                           [--plot-path PLOT_PATH]
                           prof_path database_path fail_data_path commit_sha

Run the package benchmarking.

This function runs the standard benchmarking for the pyrealm package. The profiling
tests in the test suite generate a set of combined profile data across the package
functionality. This function can then reads in a set of combined profile data and
compare it to previous benchmark data.

The profiling excludes all profiled code objects matching regex patterns provided
using the `--exclude` argument. The defaults exclude standard and site packages,
built in code and various other standard code, and are intended to reduce the
benchmarking to only code objects within the package.

positional arguments:
  prof_path              Path to pytest-profiling output
  database_path          Path to benchmarking database
  fail_data_path         Output path for data on benchmark fails
  commit_sha             Github commit SHA

options:
  -h, --help             show this help message and exit
  --exclude EXCLUDE      Exclude profiled code matching a regex pattern, can be repeated
                         (default: ['{.*}', '<.*>', '/lib/'])
  --n-runs N_RUNS        Number of most recent runs to use in benchmarking (default: 5)
  --tolerance TOLERANCE  Tolerance of time cost increase in benchmarking (default: 0.05)
  --append-on-pass       Add incoming data to the database when benchmarking passes
                         (default: False)
  --new-database         Use the incoming data to start a new profiling database
                         (default: False)
  --plot-path PLOT_PATH  Generate a benchmarking plot to this path (default: None)
```

##### Manual benchmarking

Once you have run the `pytest-profiling` test suite and generated `prof/combined.prof`
for some new code, you can run the following code to
benchmark those results.

```bash
cd profiling
poetry run python run_benchmarking.py \
       ../prof/combined.prof profiling-database.csv failure-data.csv incoming
```

This command will run the benchmark checks and print a success or failure message to the
screen. If the benchmarking fails, then the file `failure-data.csv` will be created and
will contain the incoming and database performance data for all processes that have
failed benchmarks. You can alter the benchmark tolerance and number of most recent
versions used for comparison, and also generate a plot of relative performance.

```bash
poetry run python run_benchmarking.py \
       ../prof/combined.prof profiling-database.csv failure-data.csv incoming
       --n-runs 4 --threshold 0.06 --plot-path performance-plot.png
```

If benchmarking passes, the incoming data _can_ be added to the main
`profiling-database.csv` database using the `--update-on-pass` option, but this database
should generally **only be updated by the continuous integration process** . If you do
want to compare profiles for multiple local versions of code, you can provide a new
database path and use the `--new-database` option to create a separate local database.
The `--update-on-pass` option can then be used to add data to that local file.

```bash
poetry run python run_benchmarking.py \
       ../prof/combined.prof local-database.csv failure-data.csv incoming
       --new-database
```

##### Continuous integration benchmarking

The continuous integration process runs the profiling test suite and then runs the
benchmarking with the following settings (where `8c2cbfe` is the commit SHA of the code
being profiled):

```bash
poetry run python run_benchmarking.py \
       ../prof/combined.prof profiling-database.csv failure-data.csv 8c2cbfe
        --n-runs 5 --threshold 0.05 --plot-path performance-plot.png --update-on-pass 
```

##### Resolving failed benchmarking

If benchmarking fails then the incoming code has introduced possibly troublesome
performance issues. If the code can be made more efficient, then submit commits to fix
the performance.

However, if the code cannot be made more efficient, or does something new that is
inherently more time-consuming, then the `profiling-database.csv` can be updated to
exclude performance targets that are now expected to fail. Find the rows in the database
for the most recent 5 versions for the failing code and change the `ignore_result` field
to `True` if that row sets an unacheivable target for the new code. You should also
provide a brief comment in the `ignore_justification` field to explain which commit is
being passed as a result and why.

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
