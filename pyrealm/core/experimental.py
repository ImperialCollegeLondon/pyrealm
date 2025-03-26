"""A module to provide annotation tools for experimental features.

These tools should be applied to all tools that are not guaranteed to have a stable API
within major releases. The intention is that this functionality should:

* raise a warning when experimental features are used, and
* mark experimental features clearly in the documentation.

In an ideal world, we would use a decorator to do both of these but there is not
currently a simple implementation that preserves class signatures (see
https://github.com/ImperialCollegeLondon/pyrealm/issues/429). So for the moment, this
module provides the warning and the documentation is achieved using a custom sphinx
extension that looks for an `__experimental__ = True` attribute on objects. It is easy
to set a class variable for experimental classes but for experimental functions and
methods, we need to annotate the function or method separately (and tell ``mypy`` to
shut up).

So using  `__experimental__` looks like this:

.. code:: python

    def experimental_func(a: int = 1) -> 1:
        '''A docstring.'''

        warn_experimental(experimental_func)

        return a + 1

    experimental_func.__experimental__ = True  # type: ignore[attr-defined]


    class ExperimentalMethod:

        __experimental__ = True

        def experimental_method():

            pass

        experimental_method.__experimental__ = True  # type: ignore[attr-defined]

The ``warn_experimental`` function should be called from within the method or function
or from within the ``__init__`` for class instances.
"""

from warnings import warn


class ExperimentalFeatureWarning(Warning):
    """Warn about experimental features.

    This is just a simple wrapper on the base Warning to issue clearer warnings about
    experimental features in the code.

    Arg:
        qualname: The object name to add to the warning.
    """

    def __init__(self, qualname: str) -> None:
        self.message = (
            f"Be aware that {qualname} is an experimental feature of pyrealm and "
            f"the implementation and API may change within major versions."
        )

    def __str__(self) -> str:
        return repr(self.message)


def warn_experimental(qualname: str) -> None:
    """A simple wrapper function to generate an experimental warning."""

    warn(qualname, ExperimentalFeatureWarning)


def create_experimental_docstring(qualname: str) -> str:
    """Generate an RST admonition to indicate experimental usage.

    This cannot actually be used in practice because you cannot create a docstring
    dynamically - it has to be a single fixed string literal.

    Args:
        qualname: The object name to add to the admonition.
    """

    return f"""
.. admonition:: Experimental
    :class: Important
    
    The {qualname} method or class is an experimental feature and may
    change between major releases.
    """
