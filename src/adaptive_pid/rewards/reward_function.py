"""Shaped, multi-term reward function for adaptive PID gain scheduling.

Every term here is justified in detail in ``docs/mdp_design.md`` Section 4.
This module is pure (no I/O, no MuJoCo, no RL-framework dependency) so it can
be unit tested with synthetic scalars in isolation from the environment that
calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_pid.utils.types import PIDGains, RewardWeights


@dataclass(frozen=True)
class RewardTerms:
    """Breakdown of the reward into its individual (unweighted, pre-negation)
    components. Exposed so evaluation/logging can plot per-term contributions
    over time, not just the scalar total -- essential for diagnosing *why*
    a policy is behaving a certain way during training.
    """

    tracking: float
    overshoot: float
    settling_deficit: float
    oscillation: float
    energy: float
    gain_smoothness: float
    saturation: float
    fall_penalty: float

    def total(self, weights: RewardWeights) -> float:
        if self.fall_penalty > 0:
            # A fall is catastrophic and terminal; the per-step shaped terms
            # are not additionally charged in the same step so the fall
            # penalty is unambiguous in the reward trace (see mdp_design.md
            # Section 5, "Termination").
            return -weights.fall_penalty
        return -(
            weights.w_tracking * self.tracking
            + weights.w_overshoot * self.overshoot
            + weights.w_settling * self.settling_deficit
            + weights.w_oscillation * self.oscillation
            + weights.w_energy * self.energy
            + weights.w_gain_smoothness * self.gain_smoothness
            + weights.w_saturation * self.saturation
        )


def tracking_term(error: float) -> float:
    """ISE-style tracking cost: ``e^2``. See mdp_design.md Section 4.1."""
    return error * error


def overshoot_term(theta: float, reference: float, reference_is_step: bool) -> float:
    """One-sided overshoot penalty: only penalizes error *past* the setpoint
    in the overshoot direction, and only when tracking a step-like reference
    (overshoot is not a meaningful concept while actively ramp-tracking).
    See mdp_design.md Section 4.2.
    """
    if not reference_is_step:
        return 0.0
    overshoot_amount = (theta - reference) if reference >= 0 else (reference - theta)
    # For reference >= 0, overshoot means theta > reference (went past, same
    # side); for reference < 0, overshoot means theta < reference. Using the
    # sign of the reference to determine "past" direction generalizes to
    # both positive and negative step targets.
    return max(0.0, overshoot_amount) ** 2


def settling_deficit_term(
    error: float, epsilon_settle: float, time_since_reference_change: float, expected_settle_time: float
) -> float:
    """Binary deficit: 1.0 if the plant should have settled by now but hasn't.
    See mdp_design.md Section 4.3.
    """
    if time_since_reference_change <= expected_settle_time:
        return 0.0
    return 1.0 if abs(error) > epsilon_settle else 0.0


def oscillation_term(derivative_error: float, prev_derivative_error: float | None) -> float:
    """Penalizes jerk in the error signal (change in derivative-of-error).
    See mdp_design.md Section 4.4.
    """
    if prev_derivative_error is None:
        return 0.0
    delta = derivative_error - prev_derivative_error
    return delta * delta


def energy_term(control_effort: float) -> float:
    """Quadratic control-effort / energy cost. See mdp_design.md Section 4.5."""
    return control_effort * control_effort


def gain_smoothness_term(current_gains: PIDGains, previous_gains: PIDGains) -> float:
    """Penalizes large per-step gain changes. See mdp_design.md Section 4.6."""
    delta = current_gains - previous_gains
    return delta.norm() ** 2


def saturation_term(control_effort: float, u_max: float, margin: float = 0.1) -> float:
    """Soft pre-saturation warning penalty. See mdp_design.md Section 4.7."""
    threshold = u_max * (1.0 - margin)
    return 1.0 if abs(control_effort) >= threshold else 0.0


def compute_reward(
    *,
    error: float,
    theta: float,
    reference: float,
    reference_is_step: bool,
    derivative_error: float,
    prev_derivative_error: float | None,
    control_effort: float,
    current_gains: PIDGains,
    previous_gains: PIDGains,
    u_max: float,
    epsilon_settle: float,
    time_since_reference_change: float,
    expected_settle_time: float,
    fell: bool,
    weights: RewardWeights,
) -> tuple[float, RewardTerms]:
    """Compute the full shaped reward for one control step.

    Returns ``(scalar_reward, term_breakdown)``. The breakdown is returned
    unweighted (raw per-term values) so evaluation/logging can inspect each
    term independent of the weighting choice; ``weights`` is required
    explicitly (no silent default) so a caller can never accidentally score
    an episode against the wrong reward configuration.
    """
    terms = RewardTerms(
        tracking=tracking_term(error),
        overshoot=overshoot_term(theta, reference, reference_is_step),
        settling_deficit=settling_deficit_term(
            error, epsilon_settle, time_since_reference_change, expected_settle_time
        ),
        oscillation=oscillation_term(derivative_error, prev_derivative_error),
        energy=energy_term(control_effort),
        gain_smoothness=gain_smoothness_term(current_gains, previous_gains),
        saturation=saturation_term(control_effort, u_max),
        fall_penalty=1.0 if fell else 0.0,
    )
    return terms.total(weights), terms
