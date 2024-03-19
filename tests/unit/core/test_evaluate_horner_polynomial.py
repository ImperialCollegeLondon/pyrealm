"""Test of the evaluate_horner_polynomial function."""

import numpy as np

from pyrealm.core.utilities import evaluate_horner_polynomial


def test_evaluate_horner_polynomial():
    """Test the evaluate_horner_polynomial function.

    This checks that the normal form (a + bx + cx2 + dx3) and output from the
    horner form evaluation (a + x(b + x(c + dx))) are equivalent for powers up to 10.
    """
    for deg in range(10):
        cf = np.random.normal(deg + 1)
        x = np.random.normal()
        normal = np.sum(cf * np.power(x, range(len(cf))))
        horner = evaluate_horner_polynomial(x, cf)
        assert np.allclose(normal, horner)
