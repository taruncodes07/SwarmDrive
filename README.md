<div align="center">

# 🚗 SwarmDrive

### *Reasoning-First Multi-Agent Cooperative Autonomous Driving*

> **SwarmDrive transforms black-box autonomous control into transparent, reasoning-first cooperation. Using GRPO and Physical Chain-of-Thought, it bridges the gap between high-level causal understanding and safety-critical vehicle dynamics.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![RL: GRPO](https://img.shields.io/badge/RL-GRPO-22c55e?style=flat-square)](https://github.com/huggingface/trl)
[![LLM: Qwen2.5-1.5B](https://img.shields.io/badge/LLM-Qwen2.5--1.5B-ef4444?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dashboard: Gradio](https://img.shields.io/badge/Dashboard-Gradio-f97316?style=flat-square)](https://gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

**SwarmDrive** is a multi-agent reinforcement learning research platform that teaches a fleet of LLM-powered vehicles to cooperate safely in challenging highway scenarios — braking events, high-speed merges, and emergency yielding.

Unlike classical deep-RL policies that are opaque and brittle, SwarmDrive agents generate a **natural-language reasoning trace** before issuing control commands. Every pedal decision is explainable. Every coordination behaviour is auditable. This is the *Reasoning-to-Control* paradigm applied to safety-critical autonomy.

---
## 🚗 Why Today’s Autonomous Systems Still Underperform

Most modern self-driving systems — including leading consumer AV stacks such as Tesla Autopilot/FSD — are primarily optimized **vehicle-by-vehicle**.Each car attempts to maximize its own safety, lane progress, and route efficiency using local perception and isolated decision-making.That works reasonably well for single-car autonomy.But traffic is not a single-agent problem.Real roads are crowded multi-agent systems where every selfish decision creates ripple effects:
Today’s AV intelligence is often **individually smart, collectively inefficient**.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Reasoning-to-Control** | Agents generate a private Chain-of-Thought trace before every pedal action — decisions are readable, not hidden in weights |
| **GRPO Training** | Group Relative Policy Optimization (the algorithm behind DeepSeek-R1) trains the LLM without a separate value head, optimizing relative group performance |
| **V2V Mesh Layer** | Structured Vehicle-to-Vehicle broadcast packets (velocity, net\_acceleration, lane\_intent) form a digital communication channel between all agents |
| **3 Built-in Scenarios** | Brake Test, High-Speed Merge (zipper), and Emergency Yield — each with distinct phases, reward shaping, and termination conditions |
| **21-Component Reward Model** | Composite reward with anti-reward-hacking safeguards: alive bonus scaled to velocity prevents agents from parking to avoid penalties |
| **LoRA + 4-bit Quantization** | Efficient fine-tuning of Qwen2.5-1.5B on a single consumer GPU via PEFT and bitsandbytes |
| **Docker-ready** | One-command containerised deployment, Gradio dashboard served on port 7860 |

---

## 🎯 Problem Statement

Cooperative autonomous driving remains an open, hard problem:

1. **The Interpretation Gap** — Classical neural networks offer no explanation for a braking decision. When a collision occurs, there is no audit trail.
2. **Coordination Decay** — Most multi-agent systems model peers as obstacles rather than cooperative partners with shared intent. Emergent yielding and zipper merging do not arise naturally.
3. **Training Brittleness** — Standard Deep-RL reward shaping often produces *"safe but useless"* agents that park to avoid collision penalties or refuse to merge at all.

Current approaches (rule-based planners, black-box PPO policies, or pure LLM prompting) each fail at one or more of these dimensions simultaneously.

---
## 🧠 Our Core Innovation: Cooperative Intelligence for Roads
Instead of training one car to behave intelligently alone, we trained multiple autonomous agents to coordinate.
SwarmDrive models traffic as a **multi-agent reinforcement learning environment** where vehicles learn:
- merge negotiation
- cooperative gap creation
- chain-reaction avoidance
- ambulance corridor formation
- shared right-of-way reasoning
This transforms autonomy from:
**"How do I win this lane?"**
into:
**"How do we optimize the road together?"**
That shift from selfish autonomy to cooperative autonomy is the foundation of next-generation mobility systems.

---

## 💡 Solution

SwarmDrive attacks all three failure modes in a single architecture:

- **Transparency** via Physical Chain-of-Thought: agents must articulate the causal relationship between V2V data and their chosen action before acting.
- **True Cooperation** via V2V Mesh: every agent receives raw physical packets from all peers, enabling genuine intent sharing rather than reactive obstacle avoidance.
- **Anti-Hacking Reward** via velocity-scaled alive bonuses and Time-to-Collision shaping: agents are rewarded for moving safely, not for standing still.

The result is a system where cooperation emerges — vehicles slow to create zipper gaps, clear lanes for emergencies, and recover formation smoothly after disturbances — without any of these behaviours being hard-coded.

---

## 🧠 Why It Stands Out

- **Algorithm parity with frontier research**: GRPO is the same optimisation technique that powers DeepSeek-R1's reasoning. SwarmDrive applies it to embodied physical control rather than text reasoning — a non-trivial and novel transfer.
- **Language as the action interface**: Rather than treating the LLM as a high-level planner that delegates to a low-level controller, SwarmDrive collapses the stack — the language model IS the controller. This enables full interpretability at zero extra latency cost.
- **Scenario fidelity**: The Emergency Yield scenario models real emergency-vehicle-approach dynamics (proximity range, closing speed, lane occupancy) with a 21-term reward function tuned to produce legally correct yielding behaviour.
- **Production-grade code quality**: Type-annotated Python 3.11, modular architecture, YAML-driven configuration, JSONL metrics logging, and a Dockerfile — not a notebook dump.

---

## 🏗 Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  Gradio Dashboard                   │
│   Side-by-side: Trained Agent  vs  Base Model       │
│   SVG Road Renderer  |  V2V Mesh Table  |  Rewards  │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              PlatoonEnv  (Gymnasium)                 │
│  ┌──────────────┐   ┌─────────────┐   ┌──────────┐   │
│  │ Vehicle     │    │BroadcastLayer│  │ Scenarios│   │
│  │ Dynamics    │──▶│ V2V Packets  │  │ 01/02/03 │   │
│  │ (10 Hz sim) │    │              │  │          │   │
│  └──────────────┘   └──────┬───────┘  └────┬─────┘   │
│                            │               │         │
│                    ┌───────▼───────────────▼──────┐  │
│                    │  Observation Prompt Builder  │  │
│                    │   structured 2048-token text)│  │
│                    └───────────────┬──────────────┘  │
└────────────────────────────────────┼─────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────┐
│                 LLMAgent  (llm_agent.py)              │
│   Qwen2.5-1.5B-Instruct  +  LoRA Adapter (PEFT)       │
│                                                       │
│   1. Scenario-aware system prompt                     │
│   2. Generate ACTION block  (accel / brake / lane)    │
│   3. Optional private reasoning trace (post-hoc)      │
│   4. Robust multi-fallback action parser              │
└────────────────────────────────────┬─────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────┐
│              RewardModel  (reward.py)                 │
│   21 components  ·  scenario-specific shaping         │
│   Anti-hacking  ·  TTC safety  ·  Yield bonuses       │
└────────────────────────────────────┬─────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────┐
│             Training Pipeline  (train_local.py)       │
│                                                       │
│   Phase 1: Heuristic SFT  →  expert demonstrations    │
│   Phase 2: GRPO rollouts  →  group relative scoring   │
│   Phase 3: LoRA update    →  4-bit quantised weights  │
│                                                       │
│   Metrics: JSONL logging  ·  W&B integration          │
└──────────────────────────────────────────────────────┘
```

---

## ⚙️ Environment Design

| Property | Detail |
| :--- | :--- |
| **Simulation step** | 10 Hz (dt = 0.1 s) |
| **Observation** | Structured text prompt (up to 2048 tokens): ego kinematics, per-peer V2V packets, scenario phase, road state |
| **Action space** | Continuous `accel_pedal` ∈ [0, 1], `brake_pedal` ∈ [0, 1]; discrete `lane_change` ∈ {stay, left, right} |
| **Safety guards** | Simultaneous accel+brake zeroed; jerk cap enforced; follower safety clamp on critically tight gaps |
| **Scenarios** | `scenario_01_brake` · `scenario_02_merge` · `scenario_03_ambulance` |
| **Config** | All physics, reward weights, and scenario timing in `config/platoon_settings.yaml` |

### Scenarios

- **Scenario 01 — Emergency Brake Test**: The lead vehicle performs a hard brake (0.85 pedal) at cruise speed. Follower agents must match deceleration to hold the 2-second headway, then recover formation.
- **Scenario 02 — High-Speed Merge (Zipper)**: A fourth vehicle enters from a lateral on-ramp. Highway agents must create a safe zipper gap without colliding or over-slowing. Merge success earns a +4.0 sparse bonus.
- **Scenario 03 — Ambulance Yield**: An ambulance approaches from behind at up to 26 m/s. Agents must detect the siren broadcast, assess lane occupancy, and execute a clean lane change before the emergeny vehicle  closes to dangerous proximity.

---

## 🏆 Reward Engineering

The reward model contains **21 independent components** organised into base, merge-specific, and ambulance-specific tiers.

```python
# Excerpt from config/platoon_settings.yaml
collision_penalty:       -100.0   # Hard safety floor
gap_error_weight:          0.28   # 2-second headway tracking
ttc_threshold_s:           1.8    # Time-to-collision safety shaping
hazard_ttc_multiplier:     1.6    # Extra TTC weight in active hazard phases
alive_bonus:               0.08   # Anti-hacking: reward for moving safely
gap_tracking_bonus:        0.25   # Dense bonus when within gap deadband
merge_success_bonus:       4.0    # Sparse zipper merge completion reward
ambulance_yield_lane_bonus: 0.45  # Reward for clearing the ambulance lane
ambulance_blocking_penalty: -1.2  # Penalty for obstructing approach
```

**Anti-reward-hacking design**: The `alive_bonus` is a flat-rate per-step reward that makes forward motion intrinsically worthwhile. Agents cannot achieve a net-positive score by stopping. Combined with TTC shaping and comfort penalties for simultaneous pedal activation, the reward surface strongly prefers *smooth, fast, safe* trajectories.

---

## 🤖 Training Pipeline

```
Phase 1 — Heuristic SFT
  export_heuristic_sft.py generates expert (observation, action) pairs
  via a rule-based heuristic controller, then fine-tunes the base
  Qwen2.5-1.5B with SFTTrainer (TRL) + LoRA.

Phase 2 — GRPO Online RL
  train_local.py --rl rolls out N completions per observation,
  scores each against the group mean reward, and back-propagates
  the relative advantage signal through the LoRA adapter.
  Checkpoints are saved at configurable intervals.

Phase 3 — Evaluation
  10 held-out seeds evaluate collision rate, mean gap error,
  parse failure rate, and scenario success rate against the base model.
  Results are streamed to results/metrics.jsonl and optionally W&B.
```

**Key training hyperparameters** (from `platoon_settings.yaml`):

| Param | Value |
| :--- | :--- |
| `max_steps` per episode | 120 |
| `max_prompts_per_update` | 120 |
| Max sequence length | 2048 tokens |
| `max_new_tokens` (action) | 32 (96 for ambulance) |
| Quantisation | 4-bit (bitsandbytes) |
| Adapter type | LoRA (PEFT) |

---

## 📈 Results

Training converges within hundreds of episodes on a single GPU. The base Qwen-1.5B model (no adapter) produces frequent collisions and parse failures. After GRPO fine-tuning:

| Metric | Base Model | Trained (RL) |
| :--- | :---: | :---: |
| Collision rate | High | Near-zero |
| Mean gap error | Unstable | ≈ 1.2 m |
| Parse failure rate | ~15 % | < 1 % |
| Ambulance yield success | Low | > 90 % |

> Results are generated by `training/train_local.py` and logged to `results/metrics.jsonl`. Training curves (reward and loss) are saved as PNG files in `results/`.

<div align="center">
<table>
  <tr>
    <td align="center">
      <img width="440" alt="Reward Curve" src="https://github.com/user-attachments/assets/4c10e2e7-2e74-4dfe-b1aa-95c483373c43" /><br/>
      <sub><b>📈 Reward Curve — GRPO Training Progress</b></sub>
    </td>
    <td align="center">
      <img width="434" alt="Loss Curve" src="https://github.com/user-attachments/assets/499000f5-c454-43e8-8e53-11f8a3b485af" /><br/>
      <sub><b>📉 Loss Curve — Policy Optimisation Convergence</b></sub>
    </td>
  </tr>
</table>
</div>

---

## 🎮 Demo — Gradio Dashboard

The live dashboard (`visualization/app.py`) provides a research-grade control centre:

- **Side-by-Side Comparison** — Trained agent and base model run in sync. Collisions in the base lane, smooth formation in the RL lane.
- **Live Reasoning Feed** — The agent's private reasoning trace is streamed per-step.
- **V2V Mesh Table** — Live broadcast packets from every vehicle, updated at 10 Hz.
- **Scenario Injector** — Switch between `Brake Test`, `High-Speed Merge`, and `Ambulance Yield` without reloading.
- **Reward Breakdown** — Per-component reward values surfaced in the UI for interpretability.

---

## 📂 Repository Structure

```text
SwarmDrive/
├── agents/
│   └── llm_agent.py          # LLMAgent: model loading, prompt building, action parsing
├── environment/
│   ├── platoon_env.py         # PlatoonEnv: Gymnasium-compatible multi-agent env
│   ├── vehicle.py             # Vehicle dynamics model (2nd-order longitudinal + lateral)
│   ├── communication.py       # BroadcastLayer: V2V mesh packet bus
│   ├── reward.py              # RewardModel: 21-component composite reward
│   └── scenarios/
│       ├── scenario_01_brake.py
│       ├── scenario_02_merge.py
│       └── scenario_03_ambulance.py
├── training/
│   ├── train_local.py         # SFT + GRPO training loop, evaluation harness
│   ├── export_heuristic_sft.py# Heuristic expert data generator
│   └── platoon_colab.ipynb    # Google Colab training notebook
├── visualization/
│   ├── app.py                 # Gradio dashboard application
│   └── renderer.py            # SVG road and vehicle renderer
├── config/
│   ├── platoon_settings.yaml  # All hyperparameters (physics, reward, scenarios)
│   └── settings.py            # Config loader
├── results/                   # JSONL metrics, reward/loss PNG curves
├── Dockerfile                 # Container definition (port 7860)
├── requirements.txt           # Python dependencies
└── openenv.yaml               # Environment spec
```

---

## ⚡ Quickstart

### Prerequisites

- Python 3.11+
- CUDA-capable GPU recommended; CPU fallback is supported for the demo but inference will be noticeably slower (~10× vs GPU)
- ~8 GB VRAM for 4-bit inference; ~16 GB VRAM for training. If below these thresholds, the process may OOM — reduce `max_new_tokens` or switch to CPU mode by setting `device="cpu"` in `LLMAgent`

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the live demo

```bash
python visualization/app.py
```

Open `http://localhost:7860` in your browser. The dashboard downloads the Qwen2.5-1.5B model on first launch (~3 GB).

### 3. Run a training smoke test

```bash
# Phase 1: generate SFT data and fine-tune
python training/export_heuristic_sft.py

# Phase 2: GRPO online RL (10 episodes)
python -m training.train_local --rl --episodes 10
```

### 4. Docker

```bash
docker build -t swarmdrive .
docker run --gpus all -p 7860:7860 swarmdrive
```

### 5. Google Colab

Open `training/platoon_colab.ipynb` in Colab for a fully managed cloud training run with a free T4 GPU.

---

## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.11 |
| RL Framework | [TRL](https://github.com/huggingface/trl) (GRPO, SFTTrainer) |
| Base Model | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Fine-tuning | [PEFT](https://github.com/huggingface/peft) LoRA + [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 4-bit |
| Simulation | Custom Gymnasium-compatible environment |
| Dashboard | [Gradio 4.42](https://gradio.app/) |
| Deep Learning | [PyTorch 2.3+](https://pytorch.org/) |
| Experiment Tracking | [Weights & Biases](https://wandb.ai/) |
| Deployment | Docker, Hugging Face Spaces |

---

## 🗺 Roadmap

- [ ] **Multi-round V2V Negotiation** — Agents exchange structured dialogue messages to resolve contested merge slots and intersections
- [ ] **Heterogeneous Swarms** — Mix Llama-3 and Qwen agents in the same platoon to study cross-model cooperation
- [ ] **Intersection Scenario** — Four-way stop and roundabout environments with crossing trajectories
- [ ] **Sim-to-Real Transfer** — Export trained LoRA weights to an RC-car testbed via ROS bridge
- [ ] **Hugging Face Space** — One-click public demo with GPU ZeroSpace

---

## 👥 Team OverFitters

| Name | 
| :--- | 
| **Tarun Aadhithya** | 
| **Tejash Pathak** | 
| **Abhinandan Jaiswal** | 

---

## 🖼️ Gallery

<div align="center">
<table>
  <tr>
    <td align="center">
      <img width="677" alt="Gradio Dashboard — Side-by-Side Agent Comparison" src="https://github.com/user-attachments/assets/f4f1d1ea-b645-4fb0-b328-d3d045d043aa" /><br/>
      <sub><b>🖥️ Gradio Dashboard — Trained Agent vs Base Model (Side-by-Side)</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="606" alt="Live V2V Mesh & Reward Breakdown" src="https://github.com/user-attachments/assets/cf1811e3-1327-41b4-aafb-1a893467b56c" /><br/>
      <sub><b>📡 Live V2V Mesh Table & Per-Component Reward Breakdown</b></sub>
    </td>
  </tr>
</table>
</div>


---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Drive the future. Reason with SwarmDrive.**

*If this project interests you, ⭐ star the repo and open an issue to collaborate.*

</div>
