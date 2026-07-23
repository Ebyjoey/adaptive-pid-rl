# Adaptive PID Gain Scheduling Using Deep Reinforcement Learning

> **Status: work in progress, built incrementally and documented as each module lands.**
> This section is a placeholder that will be replaced by the full README (motivation,
> architecture, installation, training, evaluation, benchmarks, results, limitations,
> future work) once the codebase is complete. See `docs/architecture.md` and
> `docs/mdp_design.md` for the design written so far.

## What's implemented so far

- [x] Repository scaffolding (`docs/`, `configs/`, `src/`, `tests/`, `ros2_ws/`, `docker/`, `.github/`)
- [x] `docs/architecture.md` — system architecture, ROS2 node graph, sequence diagram
- [x] `docs/mdp_design.md` — full MDP formalization and reward-term-by-term justification
- [x] `adaptive_pid.utils` — typed config loading + shared dataclasses + structured logging
- [x] `adaptive_pid.control.pid` — discrete PID core with clamped-integration anti-windup (27 unit tests, all passing)
- [x] `adaptive_pid.control.gain_scheduler` — the single safety-clamped entry point for RL delta-gain actions
- [x] `adaptive_pid.control.ziegler_nichols` — automated closed-loop ultimate-gain tuner (baseline #3)
- [ ] `adaptive_pid.rewards` — shaped multi-term reward function
- [ ] `adaptive_pid.envs` — MuJoCo inverted pendulum plant + domain randomization + Gymnasium wrapper
- [ ] `training/` — PPO / SAC training scripts (Stable-Baselines3)
- [ ] `evaluation/` — benchmark suite, RMSE/rise-time/settling-time/overshoot metrics, plots
- [ ] `ros2_ws/` — ROS2 Humble package with plant/PID/agent/disturbance/logging/visualization nodes
- [ ] `docker/`, `.github/workflows/` — containerization and CI
- [ ] Trained policies, benchmark tables, and result plots in `assets/`

Run the tests that exist so far:

```bash
pip install -e ".[dev]"
pytest tests/unit -v
```
