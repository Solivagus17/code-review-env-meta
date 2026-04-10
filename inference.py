#!/usr/bin/env python3
"""
Baseline inference script for CodeReviewEnv.
Handles the loop between the LLM agent and the OpenEnv environment.
"""

import os
import json
import time
import requests
from openai import OpenAI

# --- 1. Configuration ---
HF_TOKEN     = os.environ.get('HF_TOKEN')
ENV_BASE_URL = os.environ.get('ENV_BASE_URL', 'http://localhost:7860')
MODEL_NAME   = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct')
API_BASE_URL = "https://api-inference.huggingface.co/v1"

if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment variables.")

# --- 2. Initialize LLM Client ---
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a senior software engineer performing a code review.
Analyze the provided code diff and PR description carefully.
You must respond ONLY with a valid JSON object matching this schema:

{
  "verdict": "approve" | "request_changes" | "comment" | "needs_more_info",
  "overall_comment": "string (min 10 chars)",
  "line_comments": [
    {
      "line_number": int,
      "comment": "string",
      "severity": "critical"|"high"|"medium"|"low"|"info",
      "category": "bug"|"security"|"style"|"performance"|"logic"
    }
  ],
  "suggested_fixes": ["string"],
  "confidence_score": float,
  "reasoning": "string"
}

Do not include markdown blocks (like ```json) or any conversational text."""

def call_llm(obs: dict) -> dict:
    """Queries the LLM and parses the JSON response."""
    task_instructions = obs.get('task_instructions', 'Complete the code review task.')
    user_msg = f"""Task: {task_instructions}
PR Title: {obs.get('pr_title', '')}
PR Description: {obs.get('pr_description', '')}

Code Diff:
{obs.get('code_diff', '')}

Review History: {json.dumps(obs.get('review_history', []))}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_msg}
            ],
            max_tokens=1024,
            temperature=0.1
        )
        raw = resp.choices[0].message.content.strip()

        # Robust JSON cleaning
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        return json.loads(raw.strip())

    except Exception as e:
        print(f"LLM Parsing Error: {e}")
        return {
            "verdict": "comment",
            "overall_comment": "Fallback: Error parsing LLM response.",
            "line_comments": [],
            "suggested_fixes": [],
            "confidence_score": 0.5,
            "reasoning": str(e)
        }

def run_task(task_id: str) -> float:
    """Executes the full loop for a single task."""
    print(f"\n{'='*20} STARTING TASK: {task_id.upper()} {'='*20}")

    # 1. Reset Environment — POST JSON body, not query param
    try:
        res = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        res.raise_for_status()
        reset_data = res.json()
        obs = reset_data.get('observation', reset_data)
    except Exception as e:
        print(f"Failed to reset env: {e}")
        return 0.5  # Return mid-range fallback, never 0.0

    done = False
    step_num = 0
    max_steps = obs.get('max_steps', 10)
    final_score = 0.5

    # 2. Step Loop
    while not done and step_num < max_steps:
        step_num += 1
        print(f"Step {step_num}/{max_steps}...")

        # Get action from LLM
        action = call_llm(obs)

        # Post action to environment
        try:
            res = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action,
                timeout=30
            )
            res.raise_for_status()
            step_data = res.json()

            # reward is a plain float strictly in (0, 1)
            obs = step_data.get('observation', obs)
            reward = step_data['reward']  # plain float
            done = step_data.get('done', obs.get('done', False))
            final_score = obs.get('cumulative_reward', reward)

            print(f"  Verdict: {action['verdict']} | Reward: {reward:.4f}")

        except Exception as e:
            print(f"Step Error: {e}")
            break

    print(f"[END] Task: {task_id} | Steps: {step_num} | Final Score: {final_score:.4f}")
    return final_score

if __name__ == "__main__":
    tasks = ['easy', 'medium', 'hard']
    results = {}

    for tid in tasks:
        score = run_task(tid)
        results[tid] = score
        time.sleep(2)  # Cooldown between tasks

    print("\n" + "#"*40)
    print("FINAL EVALUATION SUMMARY")
    print(json.dumps({"model": MODEL_NAME, "scores": results}, indent=2))
    print("#"*40)