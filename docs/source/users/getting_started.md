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

When the above prerequisites are fullfilled, you can simply install `pyrealm` by
typing the command `pip install pyrealm` into the command line. However, it is
good practice to use a virtual environment for this, to not pollute your python
environment with packages and versions you might not need for other work.

## `pyrealm` developers

We welcome contributions to improving and extending the `pyrealm` package. The
code for `pyrealm` can be found
[on Github](https://github.com/ImperialCollegeLondon/pyrealm/). A guide how to
develop for `pyrealm` can be found in the
[`CONTRIBUTING.md` file](https://github.com/ImperialCollegeLondon/pyrealm/blob/develop/CONTRIBUTING.md).
