import os
import pytest
from pyrealm import get_data


def test_get_data():
    """Check the data path function"""
    path = get_data_path('pmodel_inputs.nc')
    assert os.path.exists(path)
    assert os.path.isfile(path)
