# MDP Design: Adaptive PID Gain Scheduling

## 1. Problem Statement

We control an inverted pendulum whose true parameters (pole mass `m`, pole length `l`, pivot friction `b`, and exogenous disturbance torque `τ_d(t)`) are unknown to the controller and vary within an episode (payload change, actuator degradation) and across episodes (domain randomization). A single fixed set of PID gains `(Kp, Ki, Kd)` cannot be simultaneously optimal across this whole distribution — this is the classical motivation for gain scheduling, normally done with hand-built lookup tables. We replace the lookup table with a learned policy that outputs *gain adjustments* conditioned on the current error dynamics, so it can interpolate/extrapolate over combinations of conditions a human table would never enumerate.

## 2. State Space

We separate the **plant state** (physical, `x = [θ, θ̇]`) from the **agent observation** (what the RL policy sees), since they are not the same thing — the agent needs error-frame and controller-internal information, not raw physical coordinates, so that the *same* policy generalizes across different reference trajectories.

Observation vector `o_t ∈ R^12`:

| # | Symbol | Description | Why it's needed |
|---|---|---|---|
| 1 | `e_t = r_t - θ_t` | tracking error | primary control objective signal |
| 2 | `ie_t = ie_{t-1} + e_t·dt` (clamped) | integral of error | lets the agent see accumulated windup risk before it happens |
| 3 | `de_t = (e_t - e_{t-1})/dt` | derivative of error | needed to judge oscillation/overshoot trend |
| 4 | `θ̇_t` | angular velocity | distinguishes "approaching setpoint fast" from "approaching slowly" — same `e_t` demands different gains |
| 5 | `u_{t-1}` | previous control effort | lets the agent see saturation before choosing new gains (avoids compounding windup) |
| 6 | `d̂_t` | disturbance estimate (from a simple velocity-residual observer, see `estimation/`) | this is the single most important feature for *anticipatory* gain scheduling — a spike in `d̂` should raise Kp/Kd preemptively |
| 7-9 | `Kp_t, Ki_t, Kd_t` | current gains | the action is a *delta*, so the policy must know its current operating point to reason about the update |
| 10 | `r_t` | reference value | needed for e.g. sign conventions, not derivable from error alone if there's noise |
| 11 | `ṙ_t` | reference rate of change | lets the agent distinguish a settled step-hold from a ramp/tracking task |
| 12 | `p_t ∈ [0,1]` | normalized in-episode progress | mild signal to help the value function reason about proximity to episode end (settling-time reward shaping is bounded-horizon) |

All 12 features are normalized (running mean/std, `VecNormalize`-compatible) before being passed to the policy network, since PPO/SAC both assume roughly unit-scale inputs for stable advantage estimation.

## 3. Action Space

`a_t = [ΔKp, ΔKi, ΔKd] ∈ [-1, 1]^3`

Applied through the `GainScheduler`:

```
K_t = clip(K_{t-1} + a_t ⊙ rate_max ⊙ dt_outer,  K_min,  K_max)
```

- `rate_max` bounds how fast gains can move per outer-loop step (physically: how fast a real gain-scheduling table would be allowed to switch, to avoid destabilizing transients from gain jumps).
- `K_min, K_max` bound absolute gain magnitude (actuator/stability safety rails — the RL agent physically cannot command an unstable or negative gain, no matter what the policy outputs, because the clamp is outside the network).
- The outer loop runs at `1/N` the rate of the inner PID loop (default `N = 10`): the PID itself must run fast enough to control the plant; gain *scheduling* is a slower, more deliberative decision and doesn't need to change every single physics tick. This also reduces the effective RL decision horizon per episode, which speeds up training.

## 4. Reward Function

Industrial PID tuning objectives are inherently multi-criteria: IAE/ISE-style tracking accuracy, minimal overshoot, fast settling, low oscillation, low energy, and *smooth* gain changes (a table that oscillates between two operating points every timestep is not a usable controller even if instantaneous reward is high). We therefore use a weighted sum of shaped, bounded terms — a decomposed reward is essential here (vs. a single sparse "did it settle" signal) because credit assignment across a 10-500 step episode with continuous gain actions is otherwise far too sparse for PPO/SAC to learn from efficiently.

```
r_t = -( w_e · e_t²                                  (tracking, ISE-style)
       + w_o · overshoot_penalty_t
       + w_s · settling_bonus_deficit_t
       + w_osc · osc_t
       + w_u · u_t²                                   (energy / control effort)
       + w_Δk · ||ΔK_t||²                              (gain smoothness)
       + w_sat · saturation_penalty_t )
```

All weights are configured in `configs/training/*.yaml` under `reward`, not hard-coded, so they can be re-tuned without touching code.

### 4.1 Tracking term — `w_e · e_t²`
Squared error (ISE) rather than `|e_t|` (IAE) because ISE penalizes large transient errors more heavily, which is exactly the behavior we want early in a step response, and its gradient is smooth everywhere (including at `e=0`), which is friendlier to PPO/SAC's Gaussian policy gradients than the non-differentiable-at-0 IAE.

### 4.2 Overshoot penalty
```
overshoot_penalty_t = relu(sign(r_t)·(θ_t - r_t))²   if a reference step is active
```
Only penalizes error *past* the setpoint in the direction of overshoot (a one-sided term) — this avoids double-penalizing the same error that the ISE term already penalizes on the approach side, and specifically shapes against the classic PID failure mode of aggressive Kp/low Kd causing ringing past the target.

### 4.3 Settling-time shaping
```
settling_bonus_deficit_t = 1  if |e_t| > ε_settle AND t > t_expected_settle
                          = 0  otherwise
```
A step-function deficit rather than continuous shaping, because settling time is fundamentally about a threshold-crossing event; per-step continuous shaping here would just duplicate the ISE term. `ε_settle` (typically 2% of the reference step, matching the classical control-engineering definition of settling time) and `t_expected_settle` are set in config.

### 4.4 Oscillation term
```
osc_t = (de_t - de_{t-1})²
```
Penalizes jerk in the error signal (second derivative), which specifically discourages hunting/ringing behavior that a naive `e_t²` term alone can under-penalize if oscillations happen to be small-amplitude but persistent.

### 4.5 Energy / control effort — `w_u · u_t²`
Standard quadratic control-effort penalty (as in LQR cost functions) — reflects real actuator power consumption `P ∝ u²` for a torque-controlled joint, and discourages the agent from solving tracking purely by cranking gains to their upper clamp.

### 4.6 Gain-smoothness — `w_Δk · ||ΔK_t||²`
This is the term that most directly encodes "a lookup table that chatters between entries is not industrially usable." Without it, a policy can satisfy tracking/overshoot objectives while thrashing gains every outer-step, which would be unacceptable in a deployed gain-scheduled controller (mechanical wear, audible motor noise, risk of exciting unmodeled resonances).

### 4.7 Saturation penalty
```
saturation_penalty_t = 1  if |u_t| >= u_max·(1-margin)  else 0
```
A soft warning penalty before hard saturation is hit, so the agent learns to avoid the actuator limit rather than just being clipped at it (clipping alone gives a flat, uninformative gradient at the boundary).

## 5. Episode Structure

- Horizon: 500 inner-loop steps (5s at `dt=0.01s`) per episode by default (`configs/env/pendulum.yaml`).
- Reference: randomly sampled per-episode from `{step, doublet, sine-sweep}` to prevent the policy from overfitting to one trajectory shape.
- Domain randomization is re-sampled at `reset()` (episode-level) and disturbances/sensor noise are additionally re-sampled within-episode (step-level) — see `docs/domain_randomization.md`.
- Termination: early termination on `|θ| > θ_fail` (pendulum falls past a physical recovery limit) with a large fixed penalty, distinct from the shaped per-step reward, so the agent unambiguously learns "falling over is catastrophic" versus "tracking imperfectly is merely costly."

## 6. Baselines Compared

| Baseline | Description |
|---|---|
| Fixed PID (best single gain set) | Grid-searched offline against the *nominal* (non-randomized) plant only |
| Manual tuning | Classical trial-and-error tuning rules (fast-Kp, slow-Ki, moderate-Kd) applied by hand, representative of what an engineer would ship without ML |
| Ziegler–Nichols (closed-loop / ultimate-gain method) | Automated, textbook-standard tuning rule; see `control/ziegler_nichols.py` |
| PPO (delta-gain policy) | On-policy, better sample efficiency stability tradeoff, no replay buffer |
| SAC (delta-gain policy) | Off-policy, typically higher asymptotic performance on continuous control, entropy-regularized exploration well-suited to smooth gain adjustment |

All are evaluated on an *identical held-out* domain-randomization distribution (different random seed range than training) — see `evaluation/benchmark.py`.
