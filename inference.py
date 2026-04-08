from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from env.environment import CloudCostEnv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# This is the official submission entrypoint used by validators.


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def _safe_action_text(action: Dict[str, Any]) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=True)


def _llm_action(
    client: Any,
    model_name: str,
    task_name: str,
    observation: Dict[str, Any],
    state_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = {
        "task": task_name,
        "instructions": [
            "You are a FinOps agent. Return one JSON action with key action_type.",
            "Allowed action_type values: query_billing, query_infra, flag_anomaly, recommend_action, write_note, submit_report.",
            "Be concise and choose only one action.",
        ],
        "observation": observation,
        "state": {
            "step": state_snapshot["step"],
            "query_count": state_snapshot["query_count"],
            "flags": state_snapshot["flags"],
        },
    }

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    action = json.loads(content)
    if "action_type" not in action:
        raise ValueError("Model output missing action_type")
    return action


def run_task(task_name: str, benchmark: str, client: Any, model_name: str) -> float:
    env = CloudCostEnv(task_name=task_name, seed=7)
    obs = env.reset(task_name=task_name, seed=7)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=benchmark, model=model_name)

    try:
        for _ in range(env.max_steps):
            action: Dict[str, Any]
            try:
                action = _llm_action(
                    client=client,
                    model_name=model_name,
                    task_name=task_name,
                    observation=obs.model_dump(),
                    state_snapshot=env.state(),
                )
            except Exception as exc:
                llm_error = str(exc).replace("\n", " ").strip() or "llm_call_failed"
                action = {"action_type": "submit_report"}
                obs, reward, done, info = env.step(action)
                rewards.append(float(reward))
                steps_taken = obs.step
                log_step(
                    step=obs.step,
                    action=_safe_action_text(action),
                    reward=float(reward),
                    done=bool(done),
                    error=llm_error,
                )
                score = float(info.get("final_score", 0.0))
                break

            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))
            steps_taken = obs.step

            last_error = info.get("last_action_error")
            if last_error is None:
                raw_error = info.get("error")
                last_error = str(raw_error) if raw_error else None

            log_step(
                step=obs.step,
                action=_safe_action_text(action),
                reward=float(reward),
                done=bool(done),
                error=last_error,
            )

            if done:
                score = float(info.get("final_score", 0.0))
                break

        if steps_taken == 0 or score == 0.0:
            _, _, _, info = env.step({"action_type": "submit_report"})
            score = float(info.get("final_score", 0.0))

        score = max(0.0, min(1.0, score))
        success = score > 0.0
        return score
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    hf_token = os.getenv("HF_TOKEN")
    _local_image_name = os.getenv("LOCAL_IMAGE_NAME")
    benchmark = os.getenv("OPENENV_BENCHMARK", "cloud-cost-anomaly-hunter")

    if OpenAI is None:
        raise RuntimeError("openai package is required for inference.py")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for submission inference")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    tasks = ["task1_zombie", "task2_spike_rca", "task3_full_audit"]
    results: List[Tuple[str, float]] = []
    for task_name in tasks:
        score = run_task(task_name=task_name, benchmark=benchmark, client=client, model_name=model_name)
        results.append((task_name, score))

    payload = {
        "scores": {name: score for name, score in results},
        "aggregate_mean": sum(score for _, score in results) / len(results),
        "model": model_name,
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

if __name__ == "__main__":
    main()
