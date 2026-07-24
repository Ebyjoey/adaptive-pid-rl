"""Integration tests for GymPIDGainEnv: verifies the wiring between plant,
PID, gain scheduler, reward, and domain randomization, as opposed to unit
tests of any single piece in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_pid.envs.gym_env import ACTION_DIM, OBS_DIM, GymPIDGainEnv
from adaptive_pid.utils.config import load_env_config

pytestmark = pytest.mark.simulation

CONFIG_PATH = "configs/env/pendulum.yaml"


def make_env(seed: int = 0) -> GymPIDGainEnv:
    cfg = load_env_config(CONFIG_PATH)
    return GymPIDGainEnv(cfg, seed=seed)


class TestGymnasiumAPICompliance:
    def test_passes_official_env_checker(self):
        env = make_env()
        check_env(env, skip_render_check=True)


class TestResetContract:
    def test_reset_returns_correctly_shaped_observation_and_info(self):
        env = make_env()
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert "plant_params" in info

    def test_reset_with_same_seed_gives_reproducible_first_observation(self):
        env1 = make_env()
        env2 = make_env()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_allclose(obs1, obs2, atol=1e-5)

    def test_reset_starts_gains_at_configured_initial_values(self):
        cfg = load_env_config(CONFIG_PATH)
        env = make_env()
        obs, _ = env.reset(seed=0)
        # obs indices 6,7,8 = kp, ki, kd (see docs/mdp_design.md observation table)
        assert obs[6] == pytest.approx(cfg.initial_gains["kp"])
        assert obs[7] == pytest.approx(cfg.initial_gains["ki"])
        assert obs[8] == pytest.approx(cfg.initial_gains["kd"])


class TestStepContract:
    def test_action_space_shape(self):
        env = make_env()
        assert env.action_space.shape == (ACTION_DIM,)

    def test_step_returns_five_tuple_with_correct_types(self):
        env = make_env()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_zn_baseline_is_stable_on_the_nominal_plant_it_was_tuned_against(self):
        """The Ziegler-Nichols gains in configs/env/pendulum.yaml were
        computed (via scripts/autotune_zn.py) against the *nominal*
        (non-randomized) plant, so held-gain regulation on that exact
        nominal plant must reach the episode horizon rather than fall."""
        from adaptive_pid.envs.domain_randomization import RandomizationRanges

        cfg = load_env_config(CONFIG_PATH)
        nominal = cfg.nominal_plant
        pinned_ranges = RandomizationRanges(
            mass_range=(nominal.mass, nominal.mass),
            length_range=(nominal.length, nominal.length),
            damping_range=(nominal.damping, nominal.damping),
            actuator_gain_range=(nominal.actuator_gain, nominal.actuator_gain),
            inertia_extra_range=(nominal.inertia_extra, nominal.inertia_extra),
            disturbance_prob_per_step=0.0,
            theta_noise_std_range=(0.0, 0.0),
            theta_dot_noise_std_range=(0.0, 0.0),
            battery_voltage_droop_range=(1.0, 1.0),
        )
        env = GymPIDGainEnv(cfg, randomization_ranges=pinned_ranges, seed=0)
        env.reset(seed=0)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 1000:
            _obs, _reward, terminated, truncated, _info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            steps += 1
        assert truncated is True
        assert terminated is False

    def test_zn_baseline_is_not_universally_robust_under_full_randomization(self):
        """This is a *documented, expected finding*, not a bug: classical
        Ziegler-Nichols closed-loop tuning is a quarter-decay-ratio design
        (deliberately somewhat aggressive/marginally stable, not
        conservative), and its gains are fixed regardless of plant
        conditions. Under this project's domain-randomization ranges (mass,
        length, damping, actuator degradation all varying simultaneously),
        it is *expected* to fail to stabilize a nontrivial fraction of
        sampled configurations -- this fragility is precisely the
        motivating gap adaptive gain scheduling is meant to close (see
        docs/mdp_design.md Section 6 and the benchmark results in
        evaluation/). We assert only a loose sanity bound here (better than
        chance, not uniformly stable), so this test documents the finding
        rather than silently asserting it away."""
        stable_count = 0
        for seed in range(10):
            env = make_env(seed=seed)
            env.reset(seed=seed)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                _obs, _reward, terminated, truncated, _info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
                steps += 1
                if steps > 1000:
                    break  # safety valve against a wiring bug causing an infinite loop
            if truncated and not terminated:
                stable_count += 1
        assert 1 <= stable_count <= 9  # neither "always stable" nor "always falls" -- both would indicate a wiring bug

    def test_reward_is_finite_every_step(self):
        env = make_env()
        env.reset(seed=1)
        for _ in range(20):
            obs, reward, terminated, truncated, _info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            assert np.isfinite(reward)
            assert np.all(np.isfinite(obs))
            if terminated or truncated:
                break

    def test_gains_move_toward_action_direction(self):
        env = make_env()
        env.reset(seed=0)
        _, _, _, _, info_before = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        kp_before = info_before["gains"].kp
        _, _, _, _, info_after = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        kp_after = info_after["gains"].kp
        assert kp_after > kp_before

    def test_falling_terminates_episode(self):
        """Driving Kd/Kp toward zero and Ki high via repeated negative
        actions on Kp should eventually destabilize the pendulum into a
        fall (terminated=True) within the episode horizon, verifying the
        theta_fail termination path is wired correctly."""
        env = make_env()
        env.reset(seed=0)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 100:
            # Repeatedly slam Kp toward its minimum -- a near-zero-gain PID
            # on an unstable inverted pendulum should not be able to hold it up.
            _obs, _reward, terminated, _truncated, _info = env.step(np.array([-1.0, 0.0, -1.0], dtype=np.float32))
            steps += 1
        assert terminated is True


class TestDeterminism:
    def test_same_seed_same_action_sequence_gives_same_trajectory(self):
        actions = [np.array([0.1, -0.1, 0.05], dtype=np.float32) for _ in range(20)]

        env1 = make_env()
        env1.reset(seed=7)
        rewards1 = []
        for a in actions:
            _, r, term, trunc, _ = env1.step(a)
            rewards1.append(r)
            if term or trunc:
                break

        env2 = make_env()
        env2.reset(seed=7)
        rewards2 = []
        for a in actions:
            _, r, term, trunc, _ = env2.step(a)
            rewards2.append(r)
            if term or trunc:
                break

        np.testing.assert_allclose(rewards1, rewards2, rtol=1e-5)
