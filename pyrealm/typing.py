"""typing.py.

This module provides a custom type hint, `Flexarray`, which is designed to \
accommodate arrays from various popular Python libraries, including NumPy, \
PyTorch, and TensorFlow.

`Flexarray` is a flexible type hint that automatically adapts to the available \
libraries in the user's environment. By default, it supports `numpy.ndarray`, but \
it can also include `torch.Tensor` and `tensorflow.Tensor` if those libraries are \
installed.

## Usage

To use the `Flexarray` type hint in your code, import it from this module:

    from pyrealm.typing import Flexarray

You can then use `Flexarray` as a type hint for function parameters, return values, \
or class attributes that are expected to be arrays or tensors from any of the supported
libraries:

    def process_array(array: Flexarray) -> None:
        '''Processes an array from any supported library.'''
        if isinstance(array, np.ndarray):
            print('Processing a NumPy array.')
        elif isinstance(array, torch.Tensor):
            print('Processing a PyTorch tensor.')
        elif isinstance(array, dask.array):
            print('Processing a Dask array.')
        else:
            raise TypeError('Unsupported array type.')

## Supported Types

- **numpy.ndarray**: The default array type.
- **torch.Tensor**: Included if PyTorch is installed.
- **tensorflow.Tensor**: Included if TensorFlow is installed.

Numpy is alsoways available as a pyrealm required package, however other array api \
compliant packages are optional.

This module is intended to simplify the handling of arrays in environments where 
different array libraries may be used, providing a unified interface for working with 
array-like data structures.
"""

import numpy as np

FlexArray = np.ndarray

try:
    import torch

    TorchArrayType = torch.Tensor
    FlexArray |= TorchArrayType
except ImportError:
    pass

try:
    import dask as da

    daskArrayType = da.array
    FlexArray |= daskArrayType
except ImportError:
    pass
