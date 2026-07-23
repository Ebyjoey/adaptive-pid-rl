from __future__ import annotations

import pytest

from adaptive_pid.envs.domain_randomization import DomainRandomizer, RandomizationRanges


class TestRandomizationRangesFromDict:
    def test_builds_with_defaults_for_missing_keys(self):
        ranges = RandomizationRanges.from_dict({})
        assert ranges.mass_range == RandomizationRanges().mass_range

    def test_overrides_specified_keys(self):
        ranges = RandomizationRanges.from_dict({"mass_range": [1.0, 2.0]})
        assert ranges.mass_range == (1.0, 2.0)

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown randomization key"):
            RandomizationRanges.from_dict({"totally_made_up_key": 5})


class TestEpisodeParamSampling:
    def test_sampled_params_are_within_configured_ranges(self):
        ranges = RandomizationRanges(
            mass_range=(0.2, 0.4),
            length_range=(0.4, 0.5),
            damping_range=(0.05, 0.1),
            actuator_gain_range=(0.9, 1.0),
            inertia_extra_range=(0.0, 0.01),
            battery_voltage_droop_range=(1.0, 1.0),  # pin to 1.0 for a clean bound check
        )
        randomizer = DomainRandomizer(ranges, seed=0)
        for _ in range(200):
            params = randomizer.sample_episode_params()
            assert 0.2 <= params.mass <= 0.4
            assert 0.4 <= params.length <= 0.5
            assert 0.05 <= params.damping <= 0.1
            assert 0.9 <= params.actuator_gain <= 1.0
            assert 0.0 <= params.inertia_extra <= 0.01

    def test_reproducible_with_fixed_seed(self):
        ranges = RandomizationRanges()
        r1 = DomainRandomizer(ranges, seed=123)
        r2 = DomainRandomizer(ranges, seed=123)
        params1 = [r1.sample_episode_params() for _ in range(5)]
        params2 = [r2.sample_episode_params() for _ in range(5)]
        for p1, p2 in zip(params1, params2):
            assert p1.mass == pytest.approx(p2.mass)
            assert p1.damping == pytest.approx(p2.damping)

    def test_different_seeds_give_different_samples(self):
        ranges = RandomizationRanges()
        r1 = DomainRandomizer(ranges, seed=1)
        r2 = DomainRandomizer(ranges, seed=2)
        p1 = r1.sample_episode_params()
        p2 = r2.sample_episode_params()
        assert p1.mass != p2.mass


class TestDisturbanceSampling:
    def test_zero_probability_never_produces_a_disturbance(self):
        ranges = RandomizationRanges(disturbance_prob_per_step=0.0)
        randomizer = DomainRandomizer(ranges, seed=0)
        events = [randomizer.maybe_sample_disturbance(current_time=t * 0.01) for t in range(500)]
        assert all(e is None for e in events)

    def test_probability_one_always_produces_a_disturbance(self):
        ranges = RandomizationRanges(disturbance_prob_per_step=1.0, disturbance_torque_range=(-1.0, 1.0))
        randomizer = DomainRandomizer(ranges, seed=0)
        events = [randomizer.maybe_sample_disturbance(current_time=t * 0.01) for t in range(50)]
        assert all(e is not None for e in events)
        assert all(-1.0 <= e.torque <= 1.0 for e in events if e is not None)


class TestSensorNoiseSampling:
    def test_stds_are_within_configured_ranges(self):
        ranges = RandomizationRanges(theta_noise_std_range=(0.001, 0.005), theta_dot_noise_std_range=(0.002, 0.01))
        randomizer = DomainRandomizer(ranges, seed=0)
        for _ in range(100):
            theta_std, theta_dot_std = randomizer.sample_sensor_noise_std()
            assert 0.001 <= theta_std <= 0.005
            assert 0.002 <= theta_dot_std <= 0.01

    def test_reseed_changes_stream(self):
        ranges = RandomizationRanges()
        randomizer = DomainRandomizer(ranges, seed=1)
        first = randomizer.sample_episode_params()
        randomizer.seed(1)
        second = randomizer.sample_episode_params()
        assert first.mass == pytest.approx(second.mass)
