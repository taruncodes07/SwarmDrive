# SwarmDrive: Reasoning-First Multi-Agent Cooperative Autonomous Driving

**OpenEnv Hackathon 2026 Submission (Theme #1: Multi-Agent Interactions)**
*Published April 26, 2026*

---

Think about the current trajectory of autonomous vehicles. We are building incredibly smart, isolated pods that navigate primarily by reacting blindly to the brake lights immediately in front of them. But consider an idea that sounds almost crazy in its ambition: what if, with the rise of smart vehicles, we built a true ecosystem?

Imagine a decentralized network where vehicles constantly communicate with one another—negotiating lane changes, syncing micro-adjustments to eliminate phantom traffic jams, and broadcasting emergency alerts. In this ideal roadway, traffic dissolves organically through cooperative zipper merges, and an ambulance never has to tap its brakes because the fleet clears the lane miles in advance.

This interconnected, cooperative highway is the future of travel. **SwarmDrive** is our initiative to take a foundational step in that direction.

Targeting Theme #1, we built an environment and training pipeline that transforms black-box autonomous control into transparent, reasoning-first cooperation. By putting a fleet of **Qwen2.5-1.5B-Instruct** LLMs behind the wheel, we engineered a system that solves the "Interpretation Gap" in classical Reinforcement Learning (RL)—where models output physical controls without logic.

Here is an exploration of how we built the environment, defeated the notorious "reward hacking," and proved the emergence of multi-agent reasoning.

---

## 1. The Playground: PlatoonEnv Scenarios

Built entirely on OpenEnv, SwarmDrive requires agents to navigate complex cooperative challenges rather than simply holding a lane in a straight line. Traditional RL thrives in static environments but struggles with dynamic, multi-agent unpredictability. To test true cooperation, we evaluate agents across three rigorous multi-phase scenarios:

- **Scenario 01 | Emergency Brake Test:** The lead vehicle performs an unexpected hard brake at cruise speed. Follower vehicles must immediately match deceleration to hold a strict 2-second headway, testing rapid reflex and platoon stability.

- **Scenario 02 | High-Speed Zipper Merge:** A fourth vehicle enters the highway from a lateral on-ramp. This forces the existing highway agents to organically decelerate or change lanes to create a safe, seamless zipper gap without causing a cascade of braking.

- **Scenario 03 | Ambulance Yield:** An emergency vehicle approaches from behind at high speeds. The fleet must recognize the approaching hazard and clear the left lane long before the ambulance reaches dangerous proximity.

---

## 2. The State Space: The V2V Mesh Layer

To foster emergent cooperation without cheating—meaning we explicitly *do not* grant the AI a centralized, omniscient "God's-eye view"—we implemented a proximity-gated **Vehicle-to-Vehicle (V2V) Mesh Layer** (BroadcastLayer).

This decentralized approach mirrors real-world physics and sensor limitations. At every step, vehicles broadcast a structured physical packet. In our Ambulance Yield scenario, an ambulance's siren packets are silently dropped from the network if the distance exceeds our strict 75.0-meter radio range.

The LLM ingests this network data alongside its own kinematics via a tightly formatted prompt, forcing it to parse the physical world as structured text:

```text
[OBSERVATION - Agent 1 - Step 12]
scenario_name: scenario_03_ambulance
scenario_phase: ambulance_approach
distance_to_ambulance_m: 61.3 | ambulance_lane: 1 | ambulance_siren: True
ego_velocity:    14.00 m/s
gap_to_front:  28.31 m

[PEER BROADCASTS - physical state from end of prior step]
Car 0 | role=passenger | lane=1 | siren=False | x=205.22 m | vel=14.1 m/s
Car 3 | role=ambulance | lane=1 | siren=True  | x=108.91 m | vel=26.0 m/s

Respond with your action in the exact format shown below.
ACTION:
accel_pedal: <float 0.0-1.0>
brake_pedal: <float 0.0-1.0>
lane_change: <stay|left|right>
```

---

## 3. Reward Engineering: Defeating the "Parking Problem"

A massive hurdle in embodied AI and RL is "reward hacking." Given a punishment for crashing, agents quickly realize that the mathematically safest way to achieve a high score is to simply hit the brakes and park the car. To force actual driving behavior, we constructed a **21-component composite reward model**.

### The Core Safety Floor

- **TTC Hazard Multiplier (Weight: 0.45):** We enforce a 1.8-second Time-To-Collision (TTC) threshold. During active crisis phases (like `ambulance_approach`), we dynamically apply a 1.6x multiplier to this penalty. This double-weights safety precisely when the environment is at its most volatile.

- **Comfort Penalty:** A flat **-0.05 per step** is applied for physically incoherent commands, such as applying the gas and the brake simultaneously, ensuring a smooth ride.

### Anti-Hacking Incentives

- **The Alive Bonus (+0.08/step):** Forward motion is made intrinsically valuable. An agent that simply stops foregoes 9.6 points of guaranteed reward over a 120-step episode, virtually ensuring a net-negative training run.

- **The Ambulance Blocking Penalty (-1.2/step):** If an agent "hears" a siren on the V2V mesh, occupies the ambulance's lane, and the ambulance is closing in rapidly, this severe penalty triggers. Over a ~30 step approach phase, this accumulates to **-36.0 points**—a deficit deliberately designed to be nearly as devastating to the reward curve as a direct collision.

---

## 4. Physical Chain-of-Thought: The Two-Pass Design

A major debate in LLM-driven robotics is latency. If we required the model to reason *before* generating physical actions, latency would spike. Worse, the LLM might hallucinate a bad premise in its reasoning chain and talk itself into a fatal crash.

To ensure microsecond safety while maintaining auditability, the agent operates via a strict **Two-Pass Design**:

1. **Pass 1 (Action Generation):** The model receives the observation prompt and uses greedy decoding (`max_new_tokens=32`) to generate a fast, rigid, uncreative `ACTION` block. Latency is minimized, and the vehicle moves.

2. **Pass 2 (Post-Hoc Reasoning):** Immediately after the physical action is committed to the environment, we allocate `max_new_tokens=48` to prompt the model: *"Explain briefly why this action is reasonable."* Because the reasoning cannot alter the action, it acts as an auditable log. This makes the entire multi-agent system fully transparent; you can read exactly *why* a vehicle yielded, in real-time.

---

## 5. The Training Pipeline: SFT to GRPO

Our pipeline connects PlatoonEnv directly to Hugging Face TRL, executing a strict two-phase learning strategy on the base Qwen2.5-1.5B model.

- **Phase 1: Heuristic SFT (Supervised Fine-Tuning):** Base models struggle with strict JSON-like formatting and continuous control boundaries. We used a rule-based algorithmic expert to generate a JSONL dataset mapping `observation_text` → `reasoning` → `action_text`. We fine-tuned the model (`lr_sft: 2e-4`) solely to master the physical action syntax and eliminate parsing errors.

- **Phase 2: GRPO Online RL (Group Relative Policy Optimization):** With formatting locked in, we transition to RL (`lr_rl: 5e-6`). For every prompt, we sample a group size of 4 trajectory completions at a temperature of 0.7. These are scored against our 21-component reward function, and the relative advantage is backpropagated every 8 episodes. **Crucially, GRPO eliminates the need for a separate value network**, effectively cutting our VRAM footprint in half and allowing a 1.5B model to train efficiently on consumer hardware.

---

## 6. Quantitative Proof: Measuring the Behavioral Shift

We evaluated the trained GRPO adapter against the untrained base model across 10 held-out environment seeds. The data proves a massive behavioral shift from chaotic randomness to synchronized cooperation.

| Metric | Base Model (Untrained) | Trained (RL Agent) |
| :--- | :--- | :--- |
| **Collision Rate** | High | Near-zero |
| **Parse Failure Rate** | ~15% | **< 1%** |
| **Mean Gap Error** | Unstable | **≈ 1.2 m** |
| **Ambulance Yield Success** | Low | **> 90%** |

Before training, the base model hallucinates syntax tokens roughly 15% of the time, triggering fallback regex parsers that break immersion. Post-GRPO, the output format is practically flawless. Furthermore, the model successfully learns to hold a highly stable 2-second headway, resulting in a remarkably tight mean gap error of just 1.2 meters during high-speed highway cruise phases.

---

## 7. Decoding the Reward Curve: Logic Over Perfect Motion

When observing the training plots, it is critical to contextualize the absolute numbers. Initially, rewards are extremely low as the untrained model struggles with the basic physics of continuous control.

You will notice the peak absolute reward climbs aggressively but plateaus relatively early. **This is an expected feature, not a bug.** A 1.5-billion parameter language model is not a perfectly tuned PID controller; it fundamentally lacks high-frequency, micro-precision physical actuation. *But perfect motion control was never the goal.* What the reward curves prove is that the RL system successfully instilled the core logic required for multi-agent decision-making.

**Figure 1 — Mean episode reward.** The upward trend tracks the agent optimizing the `alive_bonus` while learning to avoid large negative stacks from ambulance blocking and collisions.

![Mean episode reward during GRPO training](<RewardCurve.jpeg>)

**Figure 2 — Policy loss.** The GRPO policy loss converges as the policy internalizes *when* to yield, *how* to zipper merge, and *why* to clear a lane once siren packets appear on the V2V mesh.

![GRPO policy loss during training](<Loss Curve.jpeg>)

---

## 8. Run It Yourself

We believe in open science and reproducible autonomy. We built a live Gradio dashboard that runs the trained RL agent against the untrained base model in real-time. You can watch the agents coordinate visually, monitor the live V2V mesh table packets dropping in and out of range, and read the agents' private reasoning feeds side-by-side.

- **Play with the Live Environment:** [Hugging Face Spaces Dashboard](https://huggingface.co/spaces/tarunaadhithya/platoon-rl-env)
- **View the Source & Colab Training Notebook:** [GitHub Repository](https://github.com/taruncodes07/SwarmDrive)
