from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from env.environment import CloudCostEnv

app = FastAPI(title="Cloud Cost Anomaly Hunter", version="1.0.0")
env = CloudCostEnv(task_name="task1_zombie", seed=7)


class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "name": "cloud-cost-anomaly-hunter"}


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    obs = env.reset(task_name=req.task_name, seed=req.seed)
    return {"observation": obs.model_dump(), "state": env.state()}


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state()
