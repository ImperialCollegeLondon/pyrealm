"""Test of the evaluate_horner_polynomial function."""

import numpy as np

from pyrealm.core.utilities import evaluate_horner_polynomial


def test_evaluate_horner_polynomial():
    """Test the evaluate_horner_polynomial function."""
    for deg in range(10):
        cf = np.random.rand(deg + 1)
        x = np.random.rand()
        normal = np.sum(cf * np.power(x, range(len(cf))))
        horner = evaluate_horner_polynomial(x, cf)
        assert np.allclose(normal, horner)
