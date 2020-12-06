The pypmodel package
====================

These are development notes for the package, user documentation can be found at:

https://pypmodel.readthedocs.io/en/develop/pmodel.html

## Overview

This a Python 3 package intended to provide a common framework for a number of
related models of plant productivity, growth and demography.


## Code development

The codebase is developed in `git` with a repository at:

https://github.com/davidorme/pypmodel

It uses the `git flow` model for development and release. Briefly:

* All code development should happen on the general `develop` branch or on specific 
  `feature/feature_name` branches.
* Candidate release versions should be made on specific `release/x.y.z` branches
  and these are then committed to the `master` branch only after final checking.
* The `master` branch should only ever contain commits representing new release
  versions - do not work on the `master` branch.

### Continuous integration

The project uses continuous integration on the Travis platform to check that the
package is building correctly as changes are committed to Github

## Documentation

The `pypmodel` package is documented using `sphinx`. In general, the
documentation is written using  reStructuredText (`.rst`) format. The code is
documented using [Google
style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Equations in the documentation are supported using the `sphinx.ext.mathjax`
extension and this has been configured load the `mhchem`, which supports
chemical notation.

You will need to install the python packages given in `source/requirements.txt`
to build the documentation. Then, to actually build the documentation, use
`make` in the package root, which will use the `Makefile` created by
`sphinx-quickstart`. 

```bash
make html
```

### Online documentation

The documentation for the package is hosted at:

https://pypmodel.readthedocs.io/en/develop/pmodel.html

This has been configured to build commits to the `master` branch, which should
generate version specific sets of documentation.

### Referencing

The documentation uses the `sphinxcontrib-bibtex` package to support citations.
This uses Latex like citation keys in the documentation to insert references and
build a bibliography. The reference library in `source/refs.bib` needs to be
kept up to date with the literature for the project.

At present, that package uses a rather ugly citation style. I'm hoping the
`sphinx_astrorefs` package will help out, but there is currently an issue
getting that package to load.

## Testing

### Using `doctest`

The package docstrings contain `doctest` examples of code use. These are
intended to demonstrate use and to validate a reference set of inputs against
expected outputs. They do not provide extensive unit testing! To run the
docstring tests, use:

```bash
python -m doctest pypmodel/pmodel.py

``` 

The sources of the reference inputs and outputs are:

* `pmodel` module: Benjamin Stocker's `rpmodel` implementation of the P-model in R (https://github.com/stineb/rpmodel/tree/master/R)

