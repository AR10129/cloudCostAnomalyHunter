from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_file(path: str) -> Dict[str, Any]:
    p = REPO_ROOT / path
    return {"name": f"file:{path}", "ok": p.exists(), "detail": "exists" if p.exists() else "missing"}


def check_openenv_yaml() -> Dict[str, Any]:
    p = REPO_ROOT / "openenv.yaml"
    if not p.exists():
        return {"name": "openenv.yaml", "ok": False, "detail": "missing"}

    raw = p.read_text(encoding="utf-8")
    required = ["name", "version", "domain", "tasks", "observation_type", "action_type", "reward_range", "max_steps"]
    missing = [k for k in required if f"{k}:" not in raw]
    return {
        "name": "openenv.yaml",
        "ok": len(missing) == 0,
        "detail": "valid" if not missing else f"missing keys: {missing}",
    }


def check_inference_entry() -> Dict[str, Any]:
    p = REPO_ROOT / "inference.py"
    if not p.exists():
        return {"name": "inference.py", "ok": False, "detail": "missing"}
    content = p.read_text(encoding="utf-8")
    required_tokens = [
        "API_BASE_URL",
        "MODEL_NAME",
        "HF_TOKEN",
        "LOCAL_IMAGE_NAME",
        "[START] task=",
        "[STEP] step=",
        "[END] success=",
        "OpenAI",
        "reward={reward:.2f}",
        "done={done_val}",
        "error={error_val}",
    ]
    missing = [token for token in required_tokens if token not in content]
    return {
        "name": "inference.py-content",
        "ok": len(missing) == 0,
        "detail": "valid" if not missing else f"missing tokens: {missing}",
    }


def check_tasks_and_graders() -> Dict[str, Any]:
    task_files = [
        "tasks/task1_zombie.py",
        "tasks/task2_spike_rca.py",
        "tasks/task3_full_audit.py",
    ]
    missing = [t for t in task_files if not (REPO_ROOT / t).exists()]
    return {
        "name": "tasks-with-graders",
        "ok": len(missing) == 0,
        "detail": "valid" if not missing else f"missing: {missing}",
    }


def check_space_endpoints() -> Dict[str, Any]:
    p = REPO_ROOT / "app.py"
    if not p.exists():
        return {"name": "hf-space-endpoints", "ok": False, "detail": "app.py missing"}
    content = p.read_text(encoding="utf-8")
    needed = ["@app.get(\"/\")", "@app.post(\"/reset\")", "@app.post(\"/step\")", "@app.get(\"/state\")"]
    missing = [n for n in needed if n not in content]
    return {
        "name": "hf-space-endpoints",
        "ok": len(missing) == 0,
        "detail": "valid" if not missing else f"missing routes: {missing}",
    }


def main() -> None:
    checks: List[Dict[str, Any]] = []
    checks.append(check_file("Dockerfile"))
    checks.append(check_file("requirements.txt"))
    checks.append(check_file("README.md"))
    checks.append(check_openenv_yaml())
    checks.append(check_inference_entry())
    checks.append(check_tasks_and_graders())
    checks.append(check_space_endpoints())

    overall = all(c["ok"] for c in checks)
    payload = {"overall_ok": overall, "checks": checks}
    print(json.dumps(payload, indent=2))

    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
