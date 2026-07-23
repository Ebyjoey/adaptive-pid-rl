"""Ziegler-Nichols closed-loop (ultimate gain) PID tuning.

Implements the classical 1942 Ziegler-Nichols method: increase a
proportional-only controller's gain until the closed loop exhibits
sustained (non-decaying, non-growing) oscillation, record that "ultimate
gain" ``Ku`` and the oscillation period ``Tu``, then compute PID gains from
the standard lookup table. This is the textbook-standard automated tuning
baseline this project benchmarks the RL agent against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from adaptive_pid.utils.types import PIDGains


@dataclass
class ZieglerNicholsResult:
    ku: float          # ultimate gain
    tu: float          # ultimate period (s)
    gains: PIDGains    # resulting PID gains


# Standard Ziegler-Nichols "classic PID" lookup table.
_KP_FACTOR = 0.6
_TI_FACTOR = 0.5   # Ti = 0.5 * Tu  =>  Ki = Kp / Ti
_TD_FACTOR = 0.125  # Td = 0.125 * Tu =>  Kd = Kp * Td


def find_ultimate_gain(
    simulate_proportional_response: Callable[[float], np.ndarray],
    dt: float,
    kp_search_range: tuple[float, float] = (0.1, 50.0),
    kp_search_steps: int = 60,
    oscillation_tolerance: float = 0.02,
) -> tuple[float, float]:
    """Search for the ultimate gain ``Ku`` and ultimate period ``Tu``.

    Parameters
    ----------
    simulate_proportional_response:
        A callable ``f(kp) -> theta_trajectory`` that simulates the plant
        under a *proportional-only* controller with gain ``kp`` from a
        fixed initial perturbation and returns the resulting angle
        trajectory. Injected as a callable (rather than this module
        importing the MuJoCo plant directly) to keep ``control`` free of
        any dependency on ``envs``, per the architecture's one-way
        dependency rule.
    dt:
        Simulation timestep of the returned trajectory.
    kp_search_range, kp_search_steps:
        Linear search grid over candidate proportional gains.
    oscillation_tolerance:
        Relative tolerance (on peak-to-peak amplitude growth/decay ratio)
        used to classify a response as "sustained oscillation" rather than
        damped or diverging.

    Returns
    -------
    (ku, tu)
    """
    candidates = np.linspace(kp_search_range[0], kp_search_range[1], kp_search_steps)
    best_kp = candidates[0]
    best_tu = dt * 10  # fallback if nothing oscillates cleanly
    best_score = np.inf  # smaller = closer to sustained (non-decaying, non-growing) oscillation

    for kp in candidates:
        theta = simulate_proportional_response(float(kp))
        peaks = _find_peak_indices(theta)
        if len(peaks) < 3:
            continue  # not enough oscillation to measure a period/decay ratio

        peak_values = theta[peaks]
        # Ratio of successive peak amplitudes: ~1.0 indicates sustained oscillation,
        # <1 indicates decay, >1 indicates growth/instability.
        amplitude_ratios = np.abs(peak_values[1:] / (peak_values[:-1] + 1e-9))
        mean_ratio = float(np.mean(amplitude_ratios))
        score = abs(mean_ratio - 1.0)

        if score < best_score:
            best_score = score
            best_kp = float(kp)
            periods = np.diff(peaks) * dt
            best_tu = float(np.mean(periods)) if len(periods) > 0 else best_tu

        if score <= oscillation_tolerance:
            break  # good enough match for sustained oscillation; stop searching higher gains

    return best_kp, best_tu


def tune(ku: float, tu: float) -> PIDGains:
    """Compute classic Ziegler-Nichols PID gains from ``(Ku, Tu)``."""
    if ku <= 0 or tu <= 0:
        raise ValueError(f"ku and tu must be positive, got ku={ku}, tu={tu}")

    kp = _KP_FACTOR * ku
    ti = _TI_FACTOR * tu
    td = _TD_FACTOR * tu
    ki = kp / ti
    kd = kp * td
    return PIDGains(kp=kp, ki=ki, kd=kd)


def autotune(
    simulate_proportional_response: Callable[[float], np.ndarray],
    dt: float,
    **search_kwargs: float,
) -> ZieglerNicholsResult:
    """Convenience wrapper: search for ``(Ku, Tu)`` and compute gains in one call."""
    ku, tu = find_ultimate_gain(simulate_proportional_response, dt, **search_kwargs)
    gains = tune(ku, tu)
    return ZieglerNicholsResult(ku=ku, tu=tu, gains=gains)


def _find_peak_indices(signal: np.ndarray) -> np.ndarray:
    """Simple local-maxima peak finder (no scipy dependency needed for
    something this small, keeping the control/ subpackage dependency-light)."""
    if len(signal) < 3:
        return np.array([], dtype=int)
    is_peak = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    indices = np.nonzero(is_peak)[0] + 1
    return indices
