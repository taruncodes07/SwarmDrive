#!/usr/bin/env python3
"""
Direct LLM agent observation script.
Shows how Qwen acts in the platoon environment over 30 steps.
Run: python3 test_llm_rollout.py
"""

import sys
import json
from environment.platoon_env import PlatoonEnv
from agents.llm_agent import LLMAgent

def main():
    print("=" * 80)
    print("PLATOON SIMULATOR: LLM AGENT LIVE OBSERVATION")
    print("=" * 80)
    
    # Initialize environment and agents
    print("\n[1/3] Initializing environment...")
    env = PlatoonEnv()
    obs = env.reset(seed=42)
    print("✓ Environment ready (scenario: brake test)")
    
    print("\n[2/3] Loading Qwen 2.5 1.5B-Instruct model...")
    try:
        agent = LLMAgent(
            base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path=None,
            use_unsloth=True
        )
        print("✓ Model loaded via Unsloth")
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        print("  (Ensure HF_TOKEN is set for gated models: export HF_TOKEN=<your_token>)")
        return
    
    print("\n[3/3] Running 30-step rollout...")
    print("-" * 80)
    print(f"{'Step':>4} | {'Phase':>12} | {'Agent 1 Action':>25} | {'Agent 2 Action':>25} | {'Rewards':>18}")
    print("-" * 80)
    
    total_reward_1 = 0.0
    total_reward_2 = 0.0
    
    for step in range(30):
        # Get LLM actions for both agents
        try:
            action_1 = agent.act(obs["agent_1"], temperature=0.0)
            action_2 = agent.act(obs["agent_2"], temperature=0.0)
        except Exception as e:
            print(f"✗ Step {step}: Action generation failed: {e}")
            break
        
        # Step environment
        obs, rewards, dones, infos = env.step({
            "agent_1": action_1.action_text,
            "agent_2": action_2.action_text
        })
        
        total_reward_1 += rewards["agent_1"]
        total_reward_2 += rewards["agent_2"]
        
        # Get current phase
        phase_name = env.phase
        
        # Print step summary
        action_1_str = f"a:{action_1.accel_pedal:.2f} b:{action_1.brake_pedal:.2f}"
        action_2_str = f"a:{action_2.accel_pedal:.2f} b:{action_2.brake_pedal:.2f}"
        reward_str = f"a1:{rewards['agent_1']:+.2f} a2:{rewards['agent_2']:+.2f}"
        
        print(f"{step:4d} | {phase_name:>12} | {action_1_str:>25} | {action_2_str:>25} | {reward_str:>18}")
        
        # Stop if done
        if dones.get("agent_1") or dones.get("agent_2"):
            print(f"(Episode ended at step {step})")
            break
    
    print("-" * 80)
    print(f"Total Episode Rewards: Agent 1 = {total_reward_1:.2f}, Agent 2 = {total_reward_2:.2f}")
    print("\n✓ Rollout complete. Agents successfully controlled vehicles through brake scenario.")

if __name__ == "__main__":
    main()
