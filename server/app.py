import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from environment import CodeReviewEnv
from models import Action, Observation
from typing import Optional, Any

app = FastAPI(title="CodeReviewEnv")

current_env = None

# Strict bounds — all scores must be strictly inside (0, 1)
SCORE_MIN = 0.001
SCORE_MAX = 0.999

def safe_clamp(val, lo=SCORE_MIN, hi=SCORE_MAX, fallback=0.5):
    """Clamp value strictly within (0, 1). Returns fallback for NaN."""
    if val is None:
        return fallback
    if isinstance(val, dict):
        # If somehow a dict slips through, extract 'total'
        val = val.get('total', fallback)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return fallback
    if val != val:  # NaN check
        return fallback
    return max(lo, min(hi, val))

class ResetRequest(BaseModel):
    task_id: str = "easy"

@app.get("/")
def get_root():
    return {"status": "ok", "env": "CodeReviewEnv"}

@app.get("/health")
def get_health():
    return {"status": "healthy"}

@app.get("/tasks")
def get_tasks():
    return ["easy", "medium", "hard"]

@app.post("/reset")
def post_reset(req: Optional[ResetRequest] = None):
    global current_env
    try:
        if req is None:
            req = ResetRequest()
        current_env = CodeReviewEnv(task_id=req.task_id)
        obs = current_env.reset()
        obs_dict = obs.model_dump()

        # Ensure cumulative_reward is clamped
        obs_dict['cumulative_reward'] = safe_clamp(obs_dict.get('cumulative_reward', 0))

        # OpenEnv protocol: observation should contain a 'reward' field
        obs_dict['reward'] = None  # No reward at reset

        return {
            "observation": obs_dict,
            "reward": None,
            "done": False,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def post_step(action: Action):
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        action.confidence_score = safe_clamp(action.confidence_score)
        obs, reward, done, info = current_env.step(action)

        # CRITICAL: Allow the per-step reward to be a raw delta (even negative or zero).
        # We only strictly clamp cumulative episodic sums. The OpenEnv spec aggregates these delta rewards.
        reward_value = float(reward.total)

        obs_dict = obs.model_dump()
        obs_dict['cumulative_reward'] = safe_clamp(obs_dict.get('cumulative_reward', 0))

        # OpenEnv protocol: observation should contain the reward
        obs_dict['reward'] = reward_value

        info['cumulative_reward'] = safe_clamp(info.get('cumulative_reward', 0))

        return {
            "observation": obs_dict,
            "reward": reward_value,    # Delta float, unbounded per step
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    state = current_env.state()
    state['cumulative_reward'] = safe_clamp(state.get('cumulative_reward', 0))
    return state

@app.post("/action")
def get_action():
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return {
        "action": "comment",
        "observation": current_env._build_observation().model_dump(),
        "state": current_env.state()
    }

@app.post("/mcp")
def post_mcp(request: dict[str, Any]):
    return {
        "jsonrpc": "2.0",
        "id": request.get("id", 1),
        "result": {
            "status": "ok",
            "env": "CodeReviewEnv"
        }
    }

@app.get("/metadata")
def get_metadata():
    return {
        "name": "CodeReviewEnv",
        "description": "A code review environment where an agent reviews pull requests and provides feedback."
    }

@app.get("/schema")
def get_schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "done": {"type": "boolean"},
                "cumulative_reward": {"type": "number"}
            }
        }
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()