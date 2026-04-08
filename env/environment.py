from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tasks import task1_zombie, task2_spike_rca, task3_full_audit

from .data_generator import generate_task1, generate_task2, generate_task3
from .models import AnomalyFlag, BillingSummary, Observation, Recommendation, ResourceInfo
from .reward import RewardCalculator


class CloudCostEnv:
    """OpenEnv-style FinOps environment for cloud cost anomaly hunting."""

    def __init__(self, task_name: str = "task1_zombie", seed: int = 7, max_steps: int = 40) -> None:
        self.task_name = task_name
        self.seed = seed
        self.max_steps = max_steps
        self.rewarder = RewardCalculator()
        self._state: Dict[str, Any] = {}
        self.reset(task_name=task_name, seed=seed)

    def reset(self, task_name: Optional[str] = None, seed: Optional[int] = None) -> Observation:
        if task_name is not None:
            self.task_name = task_name
        if seed is not None:
            self.seed = seed

        if self.task_name == "task1_zombie":
            task_data = generate_task1(seed=self.seed)
        elif self.task_name == "task2_spike_rca":
            task_data = generate_task2(seed=self.seed)
        elif self.task_name == "task3_full_audit":
            task_data = generate_task3(seed=self.seed)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        self._state = {
            "task": task_data["task"],
            "billing_rows": task_data["billing"],
            "infra_snapshot": task_data["infra_snapshot"],
            "saas_licenses": task_data["saas_licenses"],
            "labels": task_data["labels"],
            "agent_notes": [],
            "flags": [],
            "recommendations": [],
            "query_count": 0,
            "step": 0,
            "done": False,
            "last_query_results": None,
            "last_infra_result": None,
            "report": {},
            "submission": {},
            "cumulative_reward": 0.0,
        }

        self.rewarder = RewardCalculator()
        return self._make_observation(step_reward=0.0)

    def state(self) -> Dict[str, Any]:
        return {
            "task": self._state["task"],
            "step": self._state["step"],
            "done": self._state["done"],
            "query_count": self._state["query_count"],
            "flags": self._state["flags"],
            "recommendations": self._state["recommendations"],
            "agent_notes": self._state["agent_notes"],
            "labels": self._state["labels"],
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._state["done"]:
            return self._make_observation(step_reward=0.0), 0.0, True, {"reason": "episode_done"}

        self._state["step"] += 1
        action_type = action.get("action_type")
        info: Dict[str, Any] = {}

        if action_type == "query_billing":
            info = self._query_billing(action.get("filter", {}))
        elif action_type == "query_infra":
            info = self._query_infra(action.get("resource_id", ""))
        elif action_type == "flag_anomaly":
            info = self._flag_anomaly(action)
        elif action_type == "recommend_action":
            info = self._recommend_action(action)
        elif action_type == "write_note":
            info = self._write_note(action.get("content", ""))
        elif action_type == "submit_report":
            info = self._submit_report()
            self._state["done"] = True
        else:
            info = {"error": f"unknown action_type: {action_type}"}

        if self._state["step"] >= self.max_steps:
            self._state["done"] = True
            if not self._state.get("submission"):
                self._state["submission"] = {
                    "flags": self._state["flags"],
                    "recommendations": self._state["recommendations"],
                    "report": self._state["report"],
                }

        reward, reason = self.rewarder.evaluate(
            task_name=self._state["task"],
            action_type=action_type or "unknown",
            state=self._state,
            info=info,
        )
        self._state["cumulative_reward"] += reward

        observation = self._make_observation(step_reward=reward)
        info["reward_reason"] = reason

        if self._state["done"] and "final_score" not in info:
            info["final_score"] = self._grade_submission(self._state["submission"])

        return observation, reward, self._state["done"], info

    def _billing_summary(self) -> BillingSummary:
        rows = self._state["billing_rows"]
        service_totals: Dict[str, float] = {}
        date_totals: Dict[str, float] = {}
        total = 0.0
        for row in rows:
            service = row["service"]
            day = row["date"]
            cost = float(row["cost_usd"])
            total += cost
            service_totals[service] = service_totals.get(service, 0.0) + cost
            date_totals[day] = date_totals.get(day, 0.0) + cost

        top_services = dict(sorted(service_totals.items(), key=lambda x: x[1], reverse=True)[:3])
        day_count = len(date_totals) if date_totals else 1
        daily_avg = sum(date_totals.values()) / day_count

        return BillingSummary(
            total_cost_usd=total,
            daily_avg_usd=daily_avg,
            top_services={k: float(v) for k, v in top_services.items()},
            day_count=day_count,
        )

    def _query_billing(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        out = self._state["billing_rows"]
        for key, value in filter_dict.items():
            out = [row for row in out if row.get(key) == value]

        rows = out[:50]
        self._state["query_count"] += 1
        self._state["last_query_results"] = rows
        return {"rows": rows, "count": len(out)}

    def _query_infra(self, resource_id: str) -> Dict[str, Any]:
        self._state["query_count"] += 1
        for item in self._state["infra_snapshot"]:
            if item["resource_id"] == resource_id:
                self._state["last_infra_result"] = item
                return {"resource": item}
        return {"resource": None, "error": "resource_not_found"}

    def _flag_anomaly(self, action: Dict[str, Any]) -> Dict[str, Any]:
        flag = AnomalyFlag(
            resource_id=action.get("resource_id", ""),
            anomaly_type=action.get("anomaly_type", "unknown"),
            severity=action.get("severity", "medium"),
            reasoning=action.get("reasoning", ""),
            root_cause_category=action.get("root_cause_category"),
        )
        self._state["flags"].append(flag.model_dump())

        # Task 2 grader reads from report; populate fields from actions when provided.
        if self._state["task"] == "task2_spike_rca":
            report = self._state["report"]
            if action.get("spike_start_date"):
                report["spike_start_date"] = action["spike_start_date"]
            if action.get("service"):
                report["service"] = action["service"]
            if action.get("root_cause_category"):
                report["root_cause_category"] = action["root_cause_category"]
            self._state["report"] = report

        return {"flag": flag.model_dump()}

    def _recommend_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        rec = Recommendation(
            resource_id=action.get("resource_id", ""),
            action_type=action.get("action_type_detail", "investigate"),
            estimated_saving_usd=float(action.get("estimated_saving_usd", 0.0)),
        )
        self._state["recommendations"].append(rec.model_dump())

        report = self._state["report"]
        report["estimated_saving_usd"] = rec.estimated_saving_usd
        self._state["report"] = report

        return {"recommendation": rec.model_dump()}

    def _write_note(self, content: str) -> Dict[str, Any]:
        self._state["agent_notes"].append(content)
        return {"ok": True}

    def _submit_report(self) -> Dict[str, Any]:
        self._state["submission"] = {
            "flags": self._state["flags"],
            "recommendations": self._state["recommendations"],
            "report": self._state["report"],
        }
        final_score = self._grade_submission(self._state["submission"])
        return {"final_score": final_score}

    def _grade_submission(self, submission: Dict[str, Any]) -> float:
        labels = self._state["labels"]
        if self._state["task"] == "task1_zombie":
            return task1_zombie.grade(submission, labels)
        if self._state["task"] == "task2_spike_rca":
            return task2_spike_rca.grade(submission, labels)
        return task3_full_audit.grade(submission, labels)

    def _make_observation(self, step_reward: float) -> Observation:
        infra_result = self._state.get("last_infra_result")
        infra_obj = ResourceInfo(**infra_result) if infra_result else None

        return Observation(
            step=self._state["step"],
            billing_summary=self._billing_summary(),
            recent_query_results=self._state.get("last_query_results"),
            infra_query_result=infra_obj,
            flagged_so_far=[AnomalyFlag(**f) for f in self._state["flags"]],
            step_reward=step_reward,
            done=self._state["done"],
        )
