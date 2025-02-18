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


def experimental() -> Callable:
    """Decorate an experimental object.

    This function wraps an experimental object.
    """

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
