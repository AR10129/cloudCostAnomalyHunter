from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from env.environment import CloudCostEnv

app = FastAPI(title="Cloud Cost Anomaly Hunter", version="1.0.0")
_SESSIONS: Dict[str, CloudCostEnv] = {}
_LOCK = threading.Lock()


def _get_or_create_env(session_id: str) -> CloudCostEnv:
    with _LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = CloudCostEnv(task_name="task1_zombie", seed=7)
        return _SESSIONS[session_id]


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_name: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    action: Dict[str, Any]


@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "name": "cloud-cost-anomaly-hunter"}


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    session_id = req.session_id or "default"
    env = _get_or_create_env(session_id)
    obs = env.reset(task_name=req.task_name, seed=req.seed)
    return {"session_id": session_id, "observation": obs.model_dump(), "state": env.state()}


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    session_id = req.session_id or "default"
    env = _get_or_create_env(session_id)
    obs, reward, done, info = env.step(req.action)
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = "default") -> Dict[str, Any]:
    env = _get_or_create_env(session_id)
    return env.state()
