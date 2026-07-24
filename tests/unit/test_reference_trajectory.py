from __future__ import annotations

import numpy as np
import pytest

from adaptive_pid.envs.reference_trajectory import ReferenceTrajectory, sample_reference_trajectory
from adaptive_pid.utils.types import ReferenceType


class TestStepReference:
    def test_zero_before_step_time_amplitude_after(self):
        traj = ReferenceTrajectory(ReferenceType.STEP, amplitude=2.0, period=10.0, step_change_times=(1.0,))
        r_before, _ = traj.value_and_rate(0.5, dt=0.01)
        r_after, _ = traj.value_and_rate(1.5, dt=0.01)
        assert r_before == pytest.approx(0.0)
        assert r_after == pytest.approx(2.0)

    def test_is_step_reference_true(self):
        traj = ReferenceTrajectory(ReferenceType.STEP, amplitude=1.0, period=10.0, step_change_times=(1.0,))
        assert traj.is_step_reference() is True

    def test_time_since_last_step_change(self):
        traj = ReferenceTrajectory(ReferenceType.STEP, amplitude=1.0, period=10.0, step_change_times=(1.0,))
        assert traj.time_since_last_step_change(3.0) == pytest.approx(2.0)
        assert traj.time_since_last_step_change(0.5) == pytest.approx(0.5)  # before any step: time since t=0


class TestDoubletReference:
    def test_up_then_down(self):
        traj = ReferenceTrajectory(
            ReferenceType.DOUBLET, amplitude=1.5, period=10.0, step_change_times=(1.0, 5.0)
        )
        r_pre, _ = traj.value_and_rate(0.5, dt=0.01)
        r_mid, _ = traj.value_and_rate(3.0, dt=0.01)
        r_post, _ = traj.value_and_rate(6.0, dt=0.01)
        assert r_pre == pytest.approx(0.0)
        assert r_mid == pytest.approx(1.5)
        assert r_post == pytest.approx(0.0)

    def test_requires_two_step_change_times(self):
        traj = ReferenceTrajectory(
            ReferenceType.DOUBLET, amplitude=1.0, period=10.0, step_change_times=(1.0,)
        )
        with pytest.raises(ValueError):
            traj.value_and_rate(2.0, dt=0.01)


class TestSineSweepReference:
    def test_starts_at_zero(self):
        traj = ReferenceTrajectory(ReferenceType.SINE_SWEEP, amplitude=1.0, period=2.0, step_change_times=())
        r0, _ = traj.value_and_rate(0.0, dt=0.01)
        assert r0 == pytest.approx(0.0, abs=1e-9)

    def test_bounded_by_amplitude(self):
        traj = ReferenceTrajectory(ReferenceType.SINE_SWEEP, amplitude=1.0, period=2.0, step_change_times=())
        for t in np.linspace(0, 10, 200):
            r, _ = traj.value_and_rate(float(t), dt=0.01)
            assert abs(r) <= 1.0 + 1e-9

    def test_is_step_reference_false(self):
        traj = ReferenceTrajectory(ReferenceType.SINE_SWEEP, amplitude=1.0, period=2.0, step_change_times=())
        assert traj.is_step_reference() is False


class TestRandomWalkReference:
    def test_deterministic_given_same_time(self):
        traj = ReferenceTrajectory(ReferenceType.RANDOM_WALK, amplitude=1.0, period=3.0, step_change_times=())
        r1, _ = traj.value_and_rate(1.234, dt=0.01)
        r2, _ = traj.value_and_rate(1.234, dt=0.01)
        assert r1 == pytest.approx(r2)

    def test_roughly_bounded_by_amplitude(self):
        traj = ReferenceTrajectory(ReferenceType.RANDOM_WALK, amplitude=2.0, period=3.0, step_change_times=())
        values = [traj.value_and_rate(float(t), dt=0.01)[0] for t in np.linspace(0, 20, 500)]
        assert max(abs(v) for v in values) <= 2.0 + 1e-9  # sum of sinusoid coefficients = 1.0 * amplitude


class TestSampleReferenceTrajectory:
    def test_reproducible_with_same_rng_state(self):
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        traj1 = sample_reference_trajectory(rng1, episode_seconds=5.0, max_amplitude=1.0)
        traj2 = sample_reference_trajectory(rng2, episode_seconds=5.0, max_amplitude=1.0)
        assert traj1.kind == traj2.kind
        assert traj1.amplitude == pytest.approx(traj2.amplitude)

    def test_amplitude_within_expected_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            traj = sample_reference_trajectory(rng, episode_seconds=5.0, max_amplitude=1.0)
            assert 0.3 <= traj.amplitude <= 1.0

    def test_all_reference_kinds_are_reachable(self):
        rng = np.random.default_rng(0)
        kinds_seen = {
            sample_reference_trajectory(rng, episode_seconds=5.0, max_amplitude=1.0).kind for _ in range(200)
        }
        assert kinds_seen == {
            ReferenceType.STEP,
            ReferenceType.DOUBLET,
            ReferenceType.SINE_SWEEP,
            ReferenceType.RANDOM_WALK,
        }
