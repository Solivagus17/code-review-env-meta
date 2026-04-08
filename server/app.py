import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from environment import CodeReviewEnv
from models import Action
from typing import Optional

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
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
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
    return current_env.state()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
