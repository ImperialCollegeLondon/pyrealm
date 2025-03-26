"""Experimental features."""

from typing import Any


def label_experimental_features(
    app: Any, what: Any, name: Any, obj: Any, options: Any, lines: Any
) -> None:
    """Function to add experimental notice to docstring."""
    if hasattr(obj, "__experimental__") and obj.__experimental__:
        notice = f"""


.. admonition:: Experimental
    :class: Warning

    Be aware that ``{obj.__name__}`` is an experimental feature and the API and any
    calculated values may change *between* major releases.

"""

        lines[1:1] = notice.splitlines()


def setup(app: Any) -> dict:
    """Add the function to the app."""
    app.connect("autodoc-process-docstring", label_experimental_features)
    return {"version": "0.0.1", "parallel_read_safe": True, "parallel_write_safe": True}
