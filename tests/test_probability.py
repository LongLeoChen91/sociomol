"""Unit tests for the energy model (probability.py)."""

import math
import pytest

from linker_prediction.probability import connection_probability, bending_energy_lp


class TestBendingEnergy:
    def test_zero_angle_zero_energy(self):
        """Zero bending angle should produce zero bending energy."""
        assert bending_energy_lp(L=20.0, theta=0.0, lp=50.0) == 0.0

    def test_positive_angle_positive_energy(self):
        """Non-zero bending angle should produce positive energy."""
        E = bending_energy_lp(L=20.0, theta=math.pi / 4, lp=50.0)
        assert E > 0.0


class TestConnectionProbability:
    def test_output_in_zero_one(self):
        """Probability should always be in (0, 1]."""
        p = connection_probability(L=20.0, theta=0.5, L0=20.0, lp=50.0)
        assert 0.0 < p <= 1.0

    def test_zero_length_returns_zero(self):
        """L <= 0 should return probability 0."""
        assert connection_probability(L=0.0, theta=0.5) == 0.0
        assert connection_probability(L=-1.0, theta=0.5) == 0.0

    def test_nan_theta_returns_zero(self):
        """Non-finite theta should return probability 0."""
        assert connection_probability(L=20.0, theta=float("nan")) == 0.0

    def test_none_length_returns_zero(self):
        """None L should return probability 0."""
        assert connection_probability(L=None, theta=0.5) == 0.0

    def test_shorter_length_higher_probability(self):
        """Shorter connections should have higher probability (all else equal)."""
        p_short = connection_probability(L=10.0, theta=0.3, L0=20.0, lp=50.0)
        p_long = connection_probability(L=50.0, theta=0.3, L0=20.0, lp=50.0)
        assert p_short > p_long

    def test_smaller_angle_higher_probability(self):
        """Smaller bending angles should have higher probability (all else equal)."""
        p_small = connection_probability(L=20.0, theta=0.1, L0=20.0, lp=50.0)
        p_large = connection_probability(L=20.0, theta=2.0, L0=20.0, lp=50.0)
        assert p_small > p_large

    def test_zero_weights_returns_one(self):
        """With all weights zero, probability should be exp(0) = 1.0."""
        p = connection_probability(
            L=20.0, theta=1.0, L0=20.0, lp=50.0,
            w_wlc=0.0, w_L=0.0, w_th=0.0, w_L_sq=0.0, w_th_sq=0.0,
        )
        assert abs(p - 1.0) < 1e-10
