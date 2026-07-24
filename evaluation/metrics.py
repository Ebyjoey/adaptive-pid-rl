"""Standard control-engineering performance metrics, computed from a
recorded rollout time series.

Kept as pure functions over plain arrays (no environment/SB3 dependency) so
they are directly unit-testable against synthetic, hand-computable
trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutMetrics:
    rmse: float
    rise_time: float | None  # s, time from 10% to 90% of the first step's amplitude
    settling_time: float | None  # s, time until |error| stays within settle_epsilon and never leaves again
    overshoot_pct: float  # % of step amplitude
    steady_state_error: float  # mean |error| over the final 10% of the episode
    control_effort_rms: float
    energy: float  # sum(u^2 * dt), proxy for energy consumption
    fell: bool


def compute_rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors**2))) if len(errors) > 0 else 0.0


def compute_control_effort_rms(control_signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(control_signal**2))) if len(control_signal) > 0 else 0.0


def compute_energy(control_signal: np.ndarray, dt: float) -> float:
    return float(np.sum(control_signal**2) * dt)


def compute_rise_time(
    theta: np.ndarray, times: np.ndarray, reference_amplitude: float, step_time: float
) -> float | None:
    """Time from the reference step's 10% to 90% crossing, for a step (or
    step-like) reference only. Returns ``None`` if the response never
    reaches 90% of the target (rise time is undefined in that case)."""
    if reference_amplitude == 0:
        return None
    mask = times >= step_time
    if not np.any(mask):
        return None
    post_step_theta = theta[mask]
    post_step_times = times[mask]

    target_10 = 0.1 * reference_amplitude
    target_90 = 0.9 * reference_amplitude
    # Handle negative-amplitude steps by comparing against |target| in the
    # direction of travel.
    sign = np.sign(reference_amplitude)
    normalized = post_step_theta * sign

    idx_10 = np.argmax(normalized >= abs(target_10)) if np.any(normalized >= abs(target_10)) else None
    idx_90 = np.argmax(normalized >= abs(target_90)) if np.any(normalized >= abs(target_90)) else None
    if idx_10 is None or idx_90 is None or idx_90 <= idx_10:
        return None
    return float(post_step_times[idx_90] - post_step_times[idx_10])


def compute_settling_time(
    errors: np.ndarray, times: np.ndarray, settle_epsilon: float, step_time: float = 0.0
) -> float | None:
    """First time after ``step_time`` at which ``|error|`` enters the
    ``settle_epsilon`` band and never leaves it again for the rest of the
    recorded trajectory. Returns ``None`` if it never settles."""
    mask = times >= step_time
    if not np.any(mask):
        return None
    post_errors = np.abs(errors[mask])
    post_times = times[mask]

    within_band = post_errors <= settle_epsilon
    # Find the last index where the signal is OUTSIDE the band; settling
    # time is the first timestamp strictly after that index (i.e. it never
    # leaves the band again after this point).
    outside_indices = np.nonzero(~within_band)[0]
    if len(outside_indices) == 0:
        return float(post_times[0])  # was within the band for the entire recorded window
    last_outside = outside_indices[-1]
    if last_outside + 1 >= len(post_times):
        return None  # never re-enters the band after the last excursion
    return float(post_times[last_outside + 1])


def compute_overshoot_pct(
    theta: np.ndarray, reference_amplitude: float, step_time: float, times: np.ndarray
) -> float:
    """Peak overshoot past the reference, as a percentage of the step
    amplitude. Returns 0 for non-step references (amplitude == 0 guard)."""
    if reference_amplitude == 0:
        return 0.0
    mask = times >= step_time
    if not np.any(mask):
        return 0.0
    sign = np.sign(reference_amplitude)
    post_theta = theta[mask] * sign
    peak = float(np.max(post_theta)) if len(post_theta) > 0 else 0.0
    overshoot = max(0.0, peak - abs(reference_amplitude))
    return 100.0 * overshoot / abs(reference_amplitude)


def compute_steady_state_error(errors: np.ndarray, fraction: float = 0.1) -> float:
    """Mean absolute error over the final ``fraction`` of the recorded
    trajectory (default: last 10%), the standard operational definition of
    steady-state error."""
    if len(errors) == 0:
        return 0.0
    n_tail = max(1, int(len(errors) * fraction))
    return float(np.mean(np.abs(errors[-n_tail:])))


def compute_rollout_metrics(
    *,
    times: np.ndarray,
    errors: np.ndarray,
    theta: np.ndarray,
    control_signal: np.ndarray,
    dt: float,
    settle_epsilon: float,
    reference_amplitude: float,
    step_time: float,
    fell: bool,
) -> RolloutMetrics:
    """Compute the full standard metric set for one recorded rollout."""
    return RolloutMetrics(
        rmse=compute_rmse(errors),
        rise_time=compute_rise_time(theta, times, reference_amplitude, step_time),
        settling_time=compute_settling_time(errors, times, settle_epsilon, step_time),
        overshoot_pct=compute_overshoot_pct(theta, reference_amplitude, step_time, times),
        steady_state_error=compute_steady_state_error(errors),
        control_effort_rms=compute_control_effort_rms(control_signal),
        energy=compute_energy(control_signal, dt),
        fell=fell,
    )
