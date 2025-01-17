"""Tests of the ``pyrealm.pmodel.arrhenius`` module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="args,expected",
    argvalues=[
        pytest.param(
            dict(tc=np.array([26.85]), tc_ref=25, coef=dict(simple=dict(ha=40000))),
            np.array([1.10462263]),
            id="simple",
        )
    ],
)
class TestSimpleArrhenius:
    """Test the simple Arrhenius scaling implementations."""

    def test_calculate_simple_arrhenius_factor(self, args, expected):
        """Test calculate_simple_arrhenius_factor."""
        from pyrealm.pmodel.arrhenius import calculate_simple_arrhenius_factor

        tk = args["tc"] + 273.15
        tk_ref = args["tc_ref"] + 273.15

        assert_allclose(
            calculate_simple_arrhenius_factor(
                tk=tk, tk_ref=tk_ref, ha=args["coef"]["simple"]["ha"]
            ),
            expected,
        )

    def test_SimpleArrhenius(self, args, expected):
        """Test SimpleArrhenius."""

        from pyrealm.pmodel import PModelEnvironment
        from pyrealm.pmodel.arrhenius import SimpleArrhenius

        env = PModelEnvironment(tc=args["tc"], patm=101325, vpd=400, co2=400)
        arrh = SimpleArrhenius(env=env, reference_temperature=args["tc_ref"])

        assert_allclose(
            arrh.calculate_arrhenius_factor(coefficients=args["coef"]), expected
        )


@pytest.mark.parametrize(
    argnames="args,expected",
    argvalues=[
        pytest.param(
            dict(
                tc_leaf=np.array([10]),
                tc_ref=25,
                tc_growth=np.array([10]),
                coef=dict(
                    kattge_knorr=dict(
                        ha=71513,
                        hd=200000,
                        entropy_intercept=668.39,
                        entropy_slope=-1.07,
                    )
                ),
            ),
            np.array([0.26097563]),
            id="simple",
        )
    ],
)
class TestKattgeKnorrArrhenius:
    """Test the Kattge Knorr Arrhenius scaling implementations."""

    def test_calculate_kattge_knorr_arrhenius_factor(self, args, expected):
        """Test test_calculate_kattge_knorr_arrhenius_factor."""

        from pyrealm.pmodel.arrhenius import calculate_kattge_knorr_arrhenius_factor

        tk = args["tc_leaf"] + 273.15
        tk_ref = args["tc_ref"] + 273.15
        cf = args["coef"]["kattge_knorr"]

        assert_allclose(
            calculate_kattge_knorr_arrhenius_factor(
                tk_leaf=tk,
                tk_ref=tk_ref,
                tc_growth=args["tc_growth"],
                ha=cf["ha"],
                hd=cf["hd"],
                entropy_intercept=cf["entropy_intercept"],
                entropy_slope=cf["entropy_slope"],
            ),
            expected,
        )

    def test_KattgeKnorrArrhenius(self, args, expected):
        """Test test_KattgeKnorrArrhenius."""

        from pyrealm.pmodel import PModelEnvironment
        from pyrealm.pmodel.arrhenius import KattgeKnorrArrhenius

        env = PModelEnvironment(tc=args["tc_leaf"], patm=101325, vpd=400, co2=400)
        arrh = KattgeKnorrArrhenius(env=env, reference_temperature=args["tc_ref"])

        assert_allclose(
            arrh.calculate_arrhenius_factor(coefficients=args["coef"]), expected
        )
