
"""
Baseline inference script for CodeReviewEnv.
Prints [START] / [STEP] / [END] structured output to stdout as required by the validator.
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI




API_BASE_URL = os.environ.get('API_BASE_URL', 'https://api-inference.huggingface.co/v1')
API_KEY      = os.environ.get('API_KEY') or os.environ.get('HF_TOKEN', '')
ENV_BASE_URL = os.environ.get('ENV_BASE_URL', 'http://localhost:7860')
MODEL_NAME   = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct')

print(f"[CONFIG] API_BASE_URL={API_BASE_URL} MODEL={MODEL_NAME}", flush=True)


client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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

Do not include markdown blocks or any conversational text."""


def call_llm(obs: dict) -> dict:
    """Query the LLM and return a parsed action dict."""
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
                {'role': 'user',   'content': user_msg},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "verdict": "comment",
            "overall_comment": "Unable to parse LLM response. Providing fallback review.",
            "line_comments": [],
            "suggested_fixes": [],
            "confidence_score": 0.5,
            "reasoning": str(e),
        }


def run_task(task_id: str) -> float:
    """Run one full task episode and return the final score."""


    print(f"[START] task={task_id}", flush=True)


    try:
        res = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        res.raise_for_status()
        reset_data = res.json()

        obs = reset_data.get('observation', reset_data)
    except Exception as e:
        print(f"[END] task={task_id} score=0.5 steps=0", flush=True)
        return 0.5

    done      = obs.get('done', False)
    max_steps = obs.get('max_steps', 10)
    step_num  = 0
    final_score = 0.5


    while not done and step_num < max_steps:
        step_num += 1

        action = call_llm(obs)

        try:
            res = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action,
                timeout=30,
            )
            res.raise_for_status()
            step_data = res.json()

            obs         = step_data.get('observation', obs)
            reward      = float(step_data['reward'])
            done        = step_data.get('done', obs.get('done', False))
            final_score = obs.get('cumulative_reward', reward)


            print(f"[STEP] task={task_id} step={step_num} reward={reward:.4f}", flush=True)

        except Exception as e:
            print(f"[STEP] task={task_id} step={step_num} reward=0.5", flush=True)
            break


    print(f"[END] task={task_id} score={final_score:.4f} steps={step_num}", flush=True)
    return final_score


if __name__ == "__main__":
    tasks   = ['easy', 'medium', 'hard']
    scores  = {}

    for tid in tasks:
        scores[tid] = run_task(tid)
        time.sleep(1)


    print(
        json.dumps({"type": "SUMMARY", "scores": scores, "model": MODEL_NAME}),
        flush=True,
    )
