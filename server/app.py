import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from environment import CodeReviewEnv
from models import Action, Observation
from typing import Optional, Any

app = FastAPI(title="CodeReviewEnv")

# In-memory global environment to satisfy the OpenEnv stateless API model (partially).
# In a real app we might use session IDs, but the PRD describes single-tenant usage for inference baseline.
current_env = None

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
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def post_step(action: Action):
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        obs, reward, done, info = current_env.step(action)
        reward_dict = reward.model_dump()
        reward_dict['total'] = max(0.001, min(0.999, reward_dict['total']))
        obs_dict = obs.model_dump()
        obs_dict['cumulative_reward'] = max(0.001, min(0.999, obs_dict['cumulative_reward']))
        info['cumulative_reward'] = max(0.001, min(0.999, info['cumulative_reward']))
        return {
            "observation": obs_dict,
            "reward": reward_dict,
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
    state['cumulative_reward'] = max(0.001, min(0.999, state['cumulative_reward']))
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
