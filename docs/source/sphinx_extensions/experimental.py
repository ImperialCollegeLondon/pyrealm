"""Experimental features."""

from typing import Any


def label_experimental_features(
    app: Any, what: Any, name: Any, obj: Any, options: Any, lines: Any
) -> None:
    """Function to add experimental notice to docstring."""
    if hasattr(obj, "__experimental__") and obj.__experimental__:
        # Add an "Experimental" notice at the beginning of the docstring
        notice = [
            "",
            "",
            ".. admonition:: Experimental",
            "    :class: Important",
            "    ",
            f"    The {obj.__name__} method or class is an experimental feature "
            "and may change between major releases.",
            "",
            "",
        ]

        # for line in reversed(notice):
        #     lines.insert(0, line)
        lines += notice


def setup(app: Any) -> dict:
    """Add the function to the app."""
    app.connect("autodoc-process-docstring", label_experimental_features)
    return {
        "version": "0.0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
