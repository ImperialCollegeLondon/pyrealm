# The `pyrealm` package

These are development notes for the package, user documentation can be found at:

[](https://pyrealm.readthedocs.io/)

## Overview

This a Python 3 package intended to provide a common framework for a number of
related models of plant productivity, growth and demography.

## Code development

### Package managment

The package uses `poetry` to manage dependencies, generate virtual environments for code
development and then for package building and publication. You will need to install
`poetry` to develop the code.

### Source code management

The codebase is developed in `git` with a repository at:

[](https://github.com/davidorme/pyrealm)

It uses the `git flow` model for development and release. Briefly:

* All code development should happen specific `feature/feature_name` branches.
* Pull requests (PR) should be made to merge feature branches into the `develop` branch.
* Candidate release versions should be made on specific `release/x.y.z` branches
  and these are then committed to the `master` branch only after final checking.
* The `master` branch should only ever contain commits representing new release
  versions - do not work on the `master` branch.

Both the `develop` and `master` branches are protected on GitHub to avoid accidents!

### Code quality

The project uses the `pre-commit` tool to enforce code quality. The configuration file
`.pre-commit-config.yaml` shows the details of the tool chain, but `isort`, `black`,
`flake8` and `markdownlint` are used to maintain code quality. You will need to install
`pre-commit` to develop package code, and each PR must pass the same set of checks.

### Code testing

#### Using `doctest`

The package docstrings contain `doctest` examples of code use. These are intended to
demonstrate use and to validate a reference set of inputs against expected outputs. They
do not provide extensive unit testing! To run the docstring tests, use:

```bash
python -m doctest pyrealm/pmodel.py
python -m doctest pyrealm/*.py
```

For `doctest` on warnings, see the example for `pyrealm.utilities.convert_rh_to_vpd`
which redirects the stderr to stdout to allow for the warning text to be included in the
doctest.

#### Using `pytest`

The `test` directory contains `pytest` modules to provide greater testing of different
input combinations and to check errors are raised correctly. These are the main tests
that the package is behaving as expected.

```bash
pytest
```

### Continuous integration

The project uses continuous integration via GitHub Actions to check that the package is
building correctly and that both `doctest` and `pytest` tests are passing. The status of
builds can be seen at:

[](https://github.com/davidorme/pyrealm/actions)

## Documentation

The `pyrealm` package is documented using `sphinx`, with source material in the
`docs/source` directory.

The documentation in `source` uses [Myst
Markdown](https://myst-parser.readthedocs.io/en/latest/) rather than the standard
`sphinx` reStructuredText (`.rst`) format. This is because the documentation uses the
`myst_nb` extension to `sphinx` that supports running documentation as a Jupyter
notebook: the built documentation includes examples of running code and output plots to
demonstrate the use and behaviour of the package.

The `sphinx` configuration includes the `sphinx.ext.mathjax` extension to support
mathematical notation. This has been configured to also load the `mhchem` extension,
supporting the rendering of chemical notation.

### Docstrings

The module codes uses docstrings written in the [Google
style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
Unlike the main documentation pages, the docstrings in code are written using
reStructuredText because the `autodoc` functions in `sphinx` currently rely on `rst`
inputs. This allows the function documentation to be stored alongside the code and
included simply into the documentation.

### Building the documentation

Additional python packages given in `docs/source/requirements.txt` are needed to build
the documentation. To actually build the documentation, use `make` in the package root,
which will use the `Makefile` created by `sphinx-quickstart`.

```bash
cd docs
make html
```

### Online documentation

The documentation for the package is hosted at:

[](https://pyrealm.readthedocs.io/en/develop/pmodel.html)

This has been configured to build commits to the `master` branch, which should
generate version specific sets of documentation.

### Referencing

The documentation uses the `sphinxcontrib-bibtex` package to support citations.
This uses Latex like citation keys in the documentation to insert references and
build a bibliography. The reference library in `source/refs.bib` needs to be
kept up to date with the literature for the project. The `sphinx_astrorefs` package is
used to provide an Author Date style citation format.

## Release process

Releasing a new version of the package follows the flow below:

1. Create a `release` branch from `develop` containing the new release code.
2. Check that this branch builds correctly, that the documentation builds correctly and
   that the package publishes to the `test-pypi` repository.
3. When all is well, create pull requests on GitHub to merge the `release` branch into
   both `develop` and `master`, along with a version tag for the release.
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

[](https://readthedocs.org)

which is the admin site controlling the build process. From the Versions tab, activate
the `release/new_release` branch and wait for it to build. Check the Builds tab to see
that it has built successfully and maybe check updates! If it has built succesfully, do
check pages to make sure that page code has executed successfully, and then go back to
the Versions tab and deactivate and hide the branch.

### Create pull requests into `master` and `develop`

If all is well, then two PRs need to be made on GitHub:

* The `release` branch into `master`, to bring all commits since the last release and
  any fixes on release into `master`.
* The `release` branch into `develop`, to bring any `release` fixes back into `develop`.

Once both of those have been merged, the `feature` branch can be deleted.

### Tag, build and publish

Once the `origin` repository is merged, then use `git pull` to bring `develop` and
`master` up to date on a local repo. Then, create a tag using the release version.

```sh
git checkout master
git tag <version>
git push --tags
```

The final commit on `master` is now tagged with the release version. You can add tags on
the GitHub website, but only by using the GitHub release system and we are using PyPi to
distribute package releases.

Before publishing a package to the main PyPi site for the first time, you need to set an
API token for PyPi.

```sh
poetry config pypi-token.pypi <your-token>
```

And now you can build the packages from `master` and publish.

```sh
poetry build
poetry publish
```
