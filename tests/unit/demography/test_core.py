"""Tests of core demography objects."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def test_PandasExporter():
    """Test the PandasExporter abstract base class."""

    from pyrealm.demography.core import PandasExporter

    @dataclass
    class TestClass(PandasExporter):
        """Simple test dataclass implementing the ABC."""

        array_attrs: ClassVar[tuple[str, ...]] = ("c", "d", "e")
        c: NDArray[np.float64]
        d: NDArray[np.float64]
        e: NDArray[np.float64]

    # create instance and run method
    instance = TestClass(c=np.arange(5), d=np.arange(5), e=np.arange(5))
    pandas_out = instance.to_pandas()

    # simple checks of output class and behaviour
    assert isinstance(pandas_out, pd.DataFrame)
    assert pandas_out.shape == (5, 3)
    assert np.allclose(pandas_out.sum(axis=1), np.arange(5) * 3)
    assert np.allclose(pandas_out.sum(axis=0), np.repeat(10, 3))


def test_Cohorts():
    """Test the Cohorts abstract base class."""

    from pyrealm.demography.core import Cohorts

    @dataclass
    class TestClass(Cohorts):
        """Simple test class implementing the ABC."""

        array_attrs: ClassVar[tuple[str, ...]] = ("a", "b")

        a: NDArray[np.float64]
        b: NDArray[np.float64]

    # Create instances
    t1 = TestClass(a=np.array([1, 2, 3]), b=np.array([4, 5, 6]))
    t2 = TestClass(a=np.array([4, 5, 6]), b=np.array([7, 8, 9]))

    # Add the t2 data into t1 and check the a and b attributes are extended
    t1.add_cohorts(t2)
    assert np.allclose(t1.a, np.arange(1, 7))
    assert np.allclose(t1.b, np.arange(4, 10))

    # Drop some indices and check the a and b attributes are truncated
    t1.drop_cohorts(np.array([0, 5]))
    assert np.allclose(t1.a, np.arange(2, 6))
    assert np.allclose(t1.b, np.arange(5, 9))


def test_PandasExporter_Cohorts_multiple_inheritance():
    """Test the behaviour of a class inheriting both core ABCs."""

    from pyrealm.demography.core import Cohorts, PandasExporter

    @dataclass
    class TestClass(Cohorts, PandasExporter):
        """Test class with multiple inheritance."""

        array_attrs: ClassVar[tuple[str, ...]] = ("c", "d", "e")

        n: InitVar[int]
        start_vals: InitVar[NDArray[np.int_]]

        c: NDArray[np.float64] = field(init=False)
        d: NDArray[np.int_] = field(init=False)
        e: NDArray[np.float64] = field(init=False)

        def __post_init__(self, n: int, start_vals: NDArray[np.int_]) -> None:
            self.c = np.arange(start_vals[0], start_vals[0] + n)
            self.d = np.arange(start_vals[1], start_vals[1] + n)
            self.e = np.arange(start_vals[2], start_vals[2] + n)

    # Create instances
    t1 = TestClass(n=5, start_vals=np.array([1, 2, 3]))
    t2 = TestClass(n=3, start_vals=np.array([6, 7, 8]))

    # Test dataframe properties
    t1_out = t1.to_pandas()

    # simple checks of output class and behaviour
    assert isinstance(t1_out, pd.DataFrame)
    assert t1_out.shape == (5, 3)
    assert np.allclose(t1_out.sum(axis=1), np.array([6, 9, 12, 15, 18]))
    assert np.allclose(t1_out.sum(axis=0), np.repeat(15, 3) + np.array([0, 5, 10]))

    # Add the second set and check the results via pandas
    t1.add_cohorts(t2)
    t1_out_add = t1.to_pandas()

    # simple checks of output class and behaviour
    assert isinstance(t1_out_add, pd.DataFrame)
    assert t1_out_add.shape == (8, 3)
    assert np.allclose(t1_out_add.sum(axis=1), np.array([6, 9, 12, 15, 18, 21, 24, 27]))
    assert np.allclose(t1_out_add.sum(axis=0), np.repeat(36, 3) + np.array([0, 8, 16]))

    # Drop some entries and recheck
    t1.drop_cohorts(np.array([0, 7]))
    t1_out_drop = t1.to_pandas()

    # simple checks of output class and behaviour
    assert isinstance(t1_out_drop, pd.DataFrame)
    assert t1_out_drop.shape == (6, 3)
    assert np.allclose(t1_out_drop.sum(axis=1), np.array([9, 12, 15, 18, 21, 24]))
    assert np.allclose(t1_out_drop.sum(axis=0), np.repeat(27, 3) + np.array([0, 6, 12]))
