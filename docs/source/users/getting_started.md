---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

# Getting started with `pyrealm`

This page helps you to get started with using `pyrealm` from the ground up.

## Prerequisites

### Using the command line

The following instructions expect the user to use the command line, by which we
mean, depending on the operating system, a terminal (Linux, macOS) or
PowerShell/Cygwin Bash Shell or similar (Windows).
Please see this
[Medium article](https://medium.com/@Ilaeria/command-line-scripting-options-for-mixed-os-teams-7077a849e75f)
and
[this Software Carpentries course](https://swcarpentry.github.io/shell-novice/)
if you need further instructions.

### Installing prerequisites

Before installing `pyrealm`, make sure you have python installed. Here are the
versions for

- [Windows](https://www.python.org/downloads/windows/),
- [Linux](https://www.python.org/downloads/source/) - there usually is a python
 version installed natively which should work,
- and [macOS](https://www.python.org/downloads/macos/) - you can also install
 python using Homebrew by typing `brew install python3` into the terminal.

The `pyrealm` package can be easily installed using the python `pip` package.
Mostly, your python version should come with `pip` readily available. If this
is not the case, the
[pip documentation](https://pip.pypa.io/en/stable/installation/#get-pip-py)
shows you how to install `pip`.

## Installing `pyrealm`

When the above prerequisites are fulfilled, you can simply install `pyrealm` by
typing the command `pip install pyrealm` into the command line.

However, it is good practice to use a virtual environment for this, to not
pollute your python environment with packages and versions you might not need
for other work. Detailed information on how to do this can be found on the
[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
In short, install `virtualenv`: `pip install virtualenv`, and
then you can create a virtual environment typing `python3 -m venv .venv`
(or `py -m venv .venv` on Windows) and activate it with
`source .venv/bin/activate` (`.venv\Scripts\activate` on Windows). Then do
`pip install pyrealm` and install everything else you might want to use. When
you are done with your `pyrealm` work just type `deactivate`. You can
re-activate and alter your environment anytime once it is set up.

## Running the worked examples

The easiest way to get started with using `pyrealm` is going through the worked
examples in this documentation.
Each example can be run in a [jupyter notebook](https://jupyter.org/). For this,
it can be opened using a jupyter hub cloud service such as
[Google Colab](https://colab.research.google.com/) or a locally hosted jupyter
hub. It is also straightforward to install and run it locally on your machine
following these steps:

- Install [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) by running
`pip install jupyterlab`.
- Start JupyterLab by running `jupyter lab` from the same directory where your
jupyter notebooks are located.
- Open your jupyter notebook -- have a look at the
[JupyterLab documentation](https://jupyterlab.readthedocs.io/en/stable/user/interface.html)
for a detailed description of the user interface.

There are also a number of Python packages which are necessary for running the
examples, namely `wget` to download example data, `matplotlib`, for plotting,
 `xarray` as a data structure for handling the data and `netCDF4` as file format
 in which the example data is stored.

The following code block (run from the `jupyter_notebooks` directory of
  `pyrealm`) sets everything up for running the notebooks on Linux:

```bash
pip install virtualenv
python3 -m venv .venv
source .venv/bin/activate

# install the necessary packages
pip install jupyterlab
pip install wget matplotlib xarray netCDF4

jupyter lab

```

## `pyrealm` developers

We welcome contributions to improving and extending the `pyrealm` package. The
code for `pyrealm` can be found
[on Github](https://github.com/ImperialCollegeLondon/pyrealm/). A guide how to
develop for `pyrealm` can be found in the
[`CONTRIBUTING.md` file](https://github.com/ImperialCollegeLondon/pyrealm/blob/develop/CONTRIBUTING.md).
