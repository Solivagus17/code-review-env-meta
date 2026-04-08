#!/usr/bin/env python3
"""Baseline inference script for CodeReviewEnv."""
import os
import json
import time
import requests
from openai import OpenAI
from datetime import datetime, timezone

# 1 & 2. Environment Variables Configuration
HF_TOKEN     = os.environ['HF_TOKEN']
API_BASE_URL = os.environ.get('API_BASE_URL', 'https://api-inference.huggingface.co/v1')
MODEL_NAME   = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct')
ENV_BASE_URL = os.environ.get('ENV_BASE_URL', 'http://localhost:7860')

# 3. Initialize OpenAI client
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = '''You are a senior software engineer performing code review.
For the given code diff, you must respond in valid JSON matching this schema:
{
  "verdict":         "approve"|"request_changes"|"comment"|"needs_more_info",
  "overall_comment": "string (min 10 chars)",
  "line_comments":   [{"line_number":int,"comment":"str","severity":"critical|high|medium|low|info","category":"bug|security|style|performance|logic"}],
  "suggested_fixes": ["string"],
  "confidence_score":float,
  "reasoning":       "string"
}
Respond ONLY with the JSON object. No markdown, no preamble.'''

def call_llm(obs: dict) -> dict:
    user_msg = f'''Task: {obs['task_instructions']}

PR Title: {obs['pr_title']}
PR Description: {obs['pr_description']}

Code Diff:
{obs['code_diff']}

Review history so far: {json.dumps(obs['review_history'])}
'''
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_msg}
            ],
            max_tokens=1024,
            temperature=0.2
        )
        raw = resp.choices[0].message.content.strip()
        # Clean up in case the LLM wrapped it in markdown
        if raw.startswith('```json'):
            raw = raw[7:]
        if raw.startswith('```'):
            raw = raw[3:]
        if raw.endswith('```'):
            raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        # 6. Fallback safe action for parse errors
        return {
            "verdict": "comment",
            "overall_comment": f"Unable to parse response, adding comment. Detailed error: {str(e)}",
            "line_comments": [],
            "suggested_fixes": [],
            "confidence_score": 0.5,
            "reasoning": "parse error fallback"
        }

def run_task(task_id: str) -> float:
    ts = datetime.now(timezone.utc).isoformat()
    # 5. Exact Format Requirement
    start_log = {"task_id": task_id, "model": MODEL_NAME, "timestamp": ts}
    print(f'[START] {json.dumps(start_log)}', flush=True)
    
    # Reset environment
    reset_resp = requests.post(f'{ENV_BASE_URL}/reset', json={'task_id': task_id})
    if reset_resp.status_code != 200:
        raise RuntimeError(f"Reset failed: {reset_resp.text}")
    obs = reset_resp.json()
    
    done = False
    cum_reward = 0.0
    step = 0
    max_steps = 10  # 7. Safety cap
    
    while not done and step < max_steps:
        action = call_llm(obs)
        step_resp = requests.post(f'{ENV_BASE_URL}/step', json=action)
        if step_resp.status_code != 200:
             # In case of validation error (e.g., empty comment), break loop to avoid crashing entirely
             print(f"Error calling /step: {step_resp.text}", flush=True)
             break

        result = step_resp.json()
        obs = result['observation']
        reward = result['reward']
        done = result['done']
        
        cum_reward += reward['total']
        step += 1
        
        step_log = {
            "task_id": task_id, 
            "step": step, 
            "action_verdict": action.get('verdict', 'unknown'),
            "reward": reward['total'],
            "cumulative_reward": cum_reward,
            "done": done
        }
        print(f'[STEP] {json.dumps(step_log)}', flush=True)

    end_log = {
        "task_id": task_id, 
        "total_steps": step, 
        "final_score": cum_reward, 
        "model": MODEL_NAME, 
        "status": "success"
    }
    print(f'[END] {json.dumps(end_log)}', flush=True)
    
    return cum_reward

if __name__ == '__main__':
    scores = {}
    # 8. Main block must run all 3 tasks in order
    for task in ['easy', 'medium', 'hard']:
        scores[task] = run_task(task)
        # 9. Sleep between tasks to avoid rate limiting
        time.sleep(2)
        
    summary_log = {"type": "SUMMARY", "scores": scores, "model": MODEL_NAME}
    print(json.dumps(summary_log), flush=True)
