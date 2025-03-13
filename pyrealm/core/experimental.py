"""This module provides a simple decorator for use with experimental features of the
pyrealm codebase. It automatically adds an experimental admonition to the docstring of
decorated objects and causes those function to generate an experimental warning when
called. The structure is largely taken from  the `deprecated` package, released under
the Apache 2.0 licence.
"""  # noqa: D205

# Original licence of deprecation package:
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import functools
import textwrap
import warnings
from collections.abc import Callable
from typing import Any

from deprecated.classic import (  # type: ignore[import-untyped]
    deprecated as _classic_deprecated,
)
from deprecated.sphinx import SphinxAdapter, deprecated  # type: ignore[import-untyped]


class ExperimentalFeatureWarning2(Warning):
    """Warn about experimental features.

    This is just a simple wrapper on the base Warning to issue clearer warnings about
    experimental features in the code.
    """

    def __init__(self, qualname: str) -> None:
        self.message = (
            f"Calling {qualname} is an experimental feature of pyrealm and "
            f"the implementation and API may change within major versions."
        )

    def __str__(self) -> str:
        return repr(self.message)


def experimental(obj: Callable) -> Callable:
    """Decorate an experimental object.

    This function wraps an experimental object.
    """

    @functools.wraps(obj)
    def _object_wrapper(obj: Callable) -> Callable:
        # Get the object docstring
        existing_docstring = obj.__doc__ or ""

        # Experimental note
        experimental_notice = f"""

.. admonition:: Experimental
    :class: Important
    
    The {obj.__name__} method or class is an experimental feature and may
    change between major releases.

        """

        # Split the docstring title from the rest of the body
        string_list = existing_docstring.split("\n", 1)

        if len(string_list) > 1:
            # Need to handle indentation careully to avoid breaking docstring
            # indentation
            string_list[1] = textwrap.dedent(string_list[1])

        # Insert the experimental notice below the title and close it all up.
        string_list.insert(1, experimental_notice)

        obj.__doc__ = "".join(string_list)

        @functools.wraps(obj)
        def _inner(*args: Any, **kwargs: Any) -> Callable:
            warnings.warn(obj.__qualname__, ExperimentalFeatureWarning2)
            return obj(*args, **kwargs)

        return _inner

    return _object_wrapper


def experimental2(obj: Callable) -> Callable:
    """Decorate an experimental object.

    This function wraps an experimental object.
    """

    def _object_wrapper(obj: Callable) -> Callable:
        """Decorator to mark functions or classes as experimental."""
        setattr(obj, "__experimental__", True)

        @functools.wraps(obj)
        def _inner(*args: Any, **kwargs: Any) -> Callable:
            warnings.warn(obj.__qualname__, ExperimentalFeatureWarning2)
            return obj(*args, **kwargs)

        return _inner

    return _object_wrapper


# def experimental_class(cls: Callable) -> Callable:
#     """Document an experimental class."""

#     @functools.wraps(cls, updated=())
#     class D(cls):
#         setattr(cls, "__experimental__", True)

#     return D


@experimental2
def canary(a: int = 1) -> int:
    """A canary.

    Args:
        a: A canary
    """

    return a


@experimental
def budgie(a: int = 1) -> int:
    """A budgie.

    Args:
        a: A budgie
    """

    return a


def undecorated_budgie(a: int = 1) -> int:
    """An undecoratedbudgie.

    Args:
        a: An undecorated budgie
    """

    return a


# @experimental_class
class Test:
    """A docstring for a class."""

    def __init__(self) -> None:
        self.a = 1
        """An attribute"""

    def method(self) -> None:
        """A method."""

        pass


# @experimental_class
# class Test2:
#     """A docstring for a class."""

#     def __init__(self) -> None:
#         self.a = 1
#         """An attribute"""

#     def method(self) -> None:
#         """A method."""

#         pass


def experimental_two(
    reason: str = "", version: str = "", line_length: int = 70, **kwargs: Any
) -> Any:
    """Decorator for experimental features.

    This decorator can be used to insert a "deprecated" directive
    in your function/class docstring in order to document the
    version of the project which deprecates this functionality in your library.

    :param str reason:
        Reason message which documents the deprecation in your library (can be omitted).

    :param str version:
        Version of your project which deprecates this feature.
        If you follow the `Semantic Versioning <https://semver.org/>`_,
        the version number has the format "MAJOR.MINOR.PATCH".

    :type  line_length: int
    :param line_length:
        Max line length of the directive text. If non nul, a long text is wrapped in
        several lines.

    Keyword arguments can be:

    -   "action":
        A warning filter used to activate or not the deprecation warning.
        Can be one of "error", "ignore", "always", "default", "module", or "once".
        If ``None``, empty or missing, the global filtering mechanism is used.

    -   "category":
        The warning category to use for the deprecation warning.
        By default, the category class is :class:`~DeprecationWarning`,
        you can inherit this class to define your own deprecation warning category.

    -   "extra_stacklevel":
        Number of additional stack levels to consider instrumentation rather than user
        code. With the default value of 0, the warning refers to where the class was
        instantiated or the function was called.


    :return: a decorator used to deprecate a function.

    .. versionchanged:: 1.2.13
       Change the signature of the decorator to reflect the valid use cases.

    .. versionchanged:: 1.2.15
        Add the *extra_stacklevel* parameter.
    """
    directive = kwargs.pop("directive", "warning")
    adapter_cls = kwargs.pop("adapter_cls", SphinxAdapter)
    kwargs["reason"] = reason
    kwargs["version"] = version
    kwargs["line_length"] = line_length
    return _classic_deprecated(directive=directive, adapter_cls=adapter_cls, **kwargs)


@experimental_two(version="1.0.0")
class Test1:
    """A docstring for a class."""

    def __init__(self, a: int = 1) -> None:
        self.a = a
        """An attribute"""

    def method(self) -> None:
        """A method."""

        pass


@experimental_two(version="1.0.0")
def test(a: int) -> int:
    """A docstring for a class."""

    return a + 1


@deprecated(reason="Because", version="1.0.0")
class Test2:
    """A docstring for a class."""

    def __init__(self, a: int = 1) -> None:
        self.a = a
        """An attribute"""

    def method(self) -> None:
        """A method."""

        pass


class Test3:
    """A docstring for a class."""

    def __init__(self, a: int = 1) -> None:
        self.a = a
        """An attribute"""

    def method(self) -> None:
        """A method."""

        pass
