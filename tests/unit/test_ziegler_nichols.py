from __future__ import annotations

import numpy as np
import pytest

from adaptive_pid.control import ziegler_nichols as zn


class TestTune:
    def test_gains_scale_with_ku_and_tu(self):
        gains_a = zn.tune(ku=10.0, tu=1.0)
        gains_b = zn.tune(ku=20.0, tu=1.0)
        assert gains_b.kp > gains_a.kp
        assert all(g > 0 for g in gains_a.as_array())

    def test_rejects_non_positive_inputs(self):
        with pytest.raises(ValueError):
            zn.tune(ku=0.0, tu=1.0)
        with pytest.raises(ValueError):
            zn.tune(ku=1.0, tu=-1.0)

    def test_known_reference_values(self):
        """Cross-check against the textbook Ziegler-Nichols formulas directly."""
        ku, tu = 12.0, 0.5
        gains = zn.tune(ku, tu)
        expected_kp = 0.6 * ku
        expected_ki = expected_kp / (0.5 * tu)
        expected_kd = expected_kp * (0.125 * tu)
        assert gains.kp == pytest.approx(expected_kp)
        assert gains.ki == pytest.approx(expected_ki)
        assert gains.kd == pytest.approx(expected_kd)


class TestFindUltimateGain:
    def test_finds_higher_kp_for_more_oscillatory_synthetic_plant(self):
        """Synthetic 'plant': a damped sinusoid whose decay rate is a
        deterministic function of kp, standing in for a real proportional
        closed-loop step response, so this test needs no MuJoCo dependency."""
        dt = 0.01
        t = np.arange(0, 5.0, dt)

        def simulate(kp: float) -> np.ndarray:
            # Decay rate shrinks as kp grows, crossing near-zero (sustained
            # oscillation) around kp ~= 20 by construction.
            decay = max(0.0, (20.0 - kp) * 0.05)
            freq = 2.0 + 0.05 * kp  # oscillation frequency also shifts with kp
            return np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)

        ku, tu = zn.find_ultimate_gain(simulate, dt=dt, kp_search_range=(1.0, 40.0), kp_search_steps=40)
        assert 10.0 < ku < 30.0  # near the constructed sustained-oscillation point
        assert tu > 0.0

    def test_autotune_returns_consistent_result(self):
        dt = 0.01
        t = np.arange(0, 5.0, dt)

        def simulate(kp: float) -> np.ndarray:
            decay = max(0.0, (20.0 - kp) * 0.05)
            return np.exp(-decay * t) * np.sin(2 * np.pi * 2.0 * t)

        result = zn.autotune(simulate, dt=dt, kp_search_range=(1.0, 40.0), kp_search_steps=40)
        assert result.ku > 0
        assert result.tu > 0
        assert result.gains.kp == pytest.approx(0.6 * result.ku)
