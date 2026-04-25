---
title: Platoon RL Environment
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.42.0"
python_version: "3.11"
app_file: visualization/app.py
pinned: false
---

# Platoon RL Environment
Cooperative multi-agent emergency braking benchmark where LLM agents control follower vehicles using physics broadcasts.

## Problem
Most LLMs are weak at fast causal physical reasoning under safety constraints. This project trains a shared Qwen policy to control two follower vehicles in a highway platoon while a scripted lead vehicle executes emergency braking. The training target is cooperative anticipatory control using broadcast physical state, not text messaging. The benchmark emphasizes safety, smoothness, and formation recovery.

## Environment
Observations are structured text prompts containing ego kinematics, front-car state, phase, and delayed peer broadcasts. Actions are structured text with accel and brake pedal values. Reward combines collision penalty, gap error penalty, speed matching, jerk penalty, and a steady-state recovery bonus.

| Observation Field | Meaning |
|---|---|
| ego_velocity | Ego speed in m/s |
| gap_to_front | Bumper-to-bumper gap |
| desired_gap | max(5.0, ego_velocity*2.0) |
| peer_broadcasts | Full physical state of other cars from last step |
| scenario_phase | steady, brake_event, hold_low, recovery, steady_2 |

| Action Format | Constraint |
|---|---|
| ACTION: accel_pedal, brake_pedal | Both cannot be non-zero |

## Quick Demo
Hugging Face Space: https://huggingface.co/spaces/${HF_USERNAME}/platoon-rl-env

Click Play to watch trained agents handle an emergency brake.

## Results
Training writes plots to [results/reward_curve.png](results/reward_curve.png) and [results/loss_curve.png](results/loss_curve.png).

| Model | collision_rate (10 eval episodes) | mean_gap_error_final |
|---|---:|---:|
| Untrained baseline | TBD | TBD |
| RL-trained | TBD | TBD |

## Training
Local training is Linux-first. Native Windows is unsupported for model training.

1. Install WSL2 Ubuntu 22.04.
2. Create Python 3.11 venv.
3. Install dependencies: `pip install -r requirements.txt`
4. Fill `.env` with your HF username.
5. Smoke test: `python -m environment.platoon_env --smoke-test`
6. SFT run: `python -m training.train_local --sft --epochs 1`
7. RL run: `python -m training.train_local --rl --episodes 10`

Judge rerun notebook: [training/platoon_colab.ipynb](training/platoon_colab.ipynb)

## Why It Matters
The environment tests whether LLMs can do short-horizon physical prediction using shared state, a useful capability for AV platooning and V2V coordination tasks. It also gives a reproducible benchmark for cooperative control with language model policies. This bridges simulation RL and practical coordination objectives with interpretable text prompts.

## Links
- Space URL: https://huggingface.co/spaces/${HF_USERNAME}/platoon-rl-env
- Blog URL: TODO
- Colab URL: TODO
- WandB URL: TODO

## Notes
- Author: Tarun Aadhithya
- Editable simulation knobs (max steps, velocity limits, dynamics) are in [config/platoon_settings.yaml](config/platoon_settings.yaml)
- Editable evaluation seeds and RL runtime update limits are also in [config/platoon_settings.yaml](config/platoon_settings.yaml)
- HF runtime variables are in [.env](.env)
