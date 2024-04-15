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

# Code quality and static typing

We use `pre-commit` to ensure common code standards and style, and use `mypy` to provide
static typing of the `pyrealm` codebase.

## Using `pre-commit`

As described in the [developer overview](./overview.md), `pre-commit` is installed as
part of the `pyrealm` developer dependencies and so can be set up to run simply using:

```sh
poetry run pre-commit install
poetry run pre-commit run --all-files
```

This can take a while on the first run, and when the configuration updates, as the tool
needs to install or update all the hooks that are applied to changes within a commit.
Usually the hooks only run on files changed by a particular `git commit` but using
`pre-commit run --all-files` scans the entire codebase and is a commonly used check to
make sure all is well.

### The `pre-commit` configuration

The file
[.pre-commit-config.yaml](https://github.com/ImperialCollegeLondon/pyrealm/blob/develop/.pre-commit-config.yaml)
contains the pre-commit hooks used by `pyrealm`. The configuration file contains links
to each individual hook but in overview:

`check for merge conflicts`
: Checks for remaning `git` merge conflict markers in code files.

`debug statements (python)`
: Checks for debugger imports and `breakpoint()` calls, which should not end up in
released code.

`pyupgrade`
: Updates Python syntax to the current Python 3.10 syntax.

`isort`
: Enforces a consistent sort order and formatting of package imports in modules.

`black`
: Enforces a common code formatting across the codebase. This can be irritating but it
keeps code neatly formatted and avoids code changes that are simply alternate
formatting.

`flake8`
: Checks for a very wide range of common programming issues and is also configured to
check the docstrings on code objects.

`mypy`
: Runs static typing checks to ensure that the types of function arguments, return
values are declared and are compatible.

`markdownlint`
: Checks all markdown files for common formatting issues.

### Output and configuration

When `pre-commit` runs, you may see some lines about package installation and update,
but the key information is the output below, which shows the status of the checks set up
by each hook:

```text
check for merge conflicts................................................Passed
debug statements (python)................................................Passed
pyupgrade................................................................Passed
isort....................................................................Passed
black....................................................................Passed
flake8...................................................................Passed
mypy.....................................................................Passed
markdownlint.............................................................Passed
```

### Updating `pre-commit`

The hooks used by `pre-commit` are constantly being updated to provide new features or
to update code to deal with changes in the implementation. You can update the hooks
manually using `pre-commit autoupdate`, but the configuration is regularly updated
through our GitHub Actions workflows

## MYPY

GEtting checking to shut up.
