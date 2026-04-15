"""Contains a class to record the context when defining function arguments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Context:
    """Context class to pass between functions.

    Used to define test-specific options for generating arguments, such as shapes and
    types of arrays. Also, provides the argument definition hierarchy for use in
    'defined_method_args'.

    Attributes:
        name (str): Name of the current method/function/class.
        shapes (Callable[[int, int, str], dict[str, int]]):
            Function to generate shapes for array arguments. Takes array argument index,
            number of array arguments, and function name. Returns dictionary of
            {dimension names: dimension sizes}.
        array_type (str, optional): The type to initialise arrays as. "xarray" or
            "numpy". Defaults to "numpy".
        n_array (int): Number of array arguments in the current function.
        i_array (int): Index of the current array argument, used to select a shape.
        parents (list[str]): Hierarchy of names leading to the current function/class.
            The first value will be the name of the callable being tested. If it is a
            class method, the second value will be the name of the class.
    """

    name: str
    shapes: Callable[[int, int, str], dict[str, int]]
    array_type: str = "numpy"
    n_array: int = 0
    i_array: int = 0
    parents: list[str] = field(default_factory=list)
    """A list of the superior function / classes for the current context."""

    def new(self, name: str) -> Context:
        """Generate a context for a new function/class, updating the hierarchy."""
        return Context(
            name=name,
            shapes=self.shapes,
            array_type=self.array_type,
            n_array=self.n_array,
            i_array=self.i_array,
            parents=[*self.parents, self.name],
        )

    def _get_shape(self):
        if self.i_array == -1 or self.i_array >= self.n_array:
            # Placeholder / Fallback
            array_shape = self.shapes(0, 1, "")
        else:
            array_shape = self.shapes(self.i_array, self.n_array, self.name)
        return array_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape for index `i_array`."""
        return tuple(self._get_shape().values())

    @property
    def array_dims(self) -> tuple[str, ...]:
        """Return the dimensions for index `i_array`."""
        return tuple(self._get_shape().keys())

    @property
    def bcast_shape(self) -> tuple[int, ...]:
        """The broadcast shape of all inputs (not the full shape being tested)."""
        if self.n_array == 0:
            # Placeholder / Fallback
            return self.shape

        shapes = [self.shapes(i, self.n_array, self.name) for i in range(self.n_array)]

        # Get the full list of dimension names
        full_dims = []
        for shape in shapes:
            for dim in shape:
                if dim not in full_dims:
                    full_dims.append(dim)
        # Expand/reorder all shapes to match the full dimensions
        full_shapes = []
        for shape in shapes:
            full_shape = tuple(shape.get(dim, 1) for dim in full_dims)
            full_shapes.append(full_shape)
        # Get the full broadcast shape
        return np.broadcast_shapes(*full_shapes)
