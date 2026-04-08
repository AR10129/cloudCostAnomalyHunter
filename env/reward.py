from __future__ import annotations

from typing import Any, Dict, List, Tuple


class RewardCalculator:
    def __init__(self) -> None:
        self.cumulative = 0.0

    @staticmethod
    def _severity_multiplier(severity: str) -> float:
        return {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.5,
        }.get(severity, 0.6)

    def evaluate(
        self,
        task_name: str,
        action_type: str,
        state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> Tuple[float, str]:
        reward = 0.0
        reason = "no_change"

        if action_type in {"query_billing", "query_infra"}:
            reward += 0.005
            reason = "useful_query"

        if action_type == "write_note":
            reward += 0.002
            reason = "note_written"

        if action_type == "flag_anomaly":
            flag = info.get("flag", {})
            rid = flag.get("resource_id")
            labels = state["labels"]

            if task_name == "task1_zombie":
                if rid in labels["zombies"]:
                    reward += 0.24 * self._severity_multiplier(flag.get("severity", "medium"))
                    reason = "true_positive"
                else:
                    reward -= 0.05
                    reason = "false_positive"

            elif task_name == "task2_spike_rca":
                if rid == labels["resource_id"]:
                    reward += 0.20
                    reason = "spike_resource_identified"
                    if flag.get("root_cause_category") == labels["root_cause_category"]:
                        reward += 0.10
                        reason = "root_cause_match"
                else:
                    reward -= 0.05
                    reason = "false_positive"

            elif task_name == "task3_full_audit":
                anomalies = labels["anomalies"]
                matched = None
                for anomaly_type, payload in anomalies.items():
                    if payload["resource_id"] == rid:
                        matched = (anomaly_type, payload)
                        break

                if matched:
                    anomaly_type, payload = matched
                    sev_weight = labels["severity_weights"][payload["severity"]]
                    reward += 0.70 * sev_weight
                    if flag.get("anomaly_type") == anomaly_type:
                        reward += 0.30 * sev_weight
                        reason = "detection_plus_classification"
                    else:
                        reason = "detection_only"
                else:
                    reward -= 0.05
                    reason = "false_positive"

        if action_type == "recommend_action" and task_name in {"task2_spike_rca", "task3_full_audit"}:
            rec = info.get("recommendation", {})
            labels = state["labels"]
            if task_name == "task2_spike_rca":
                est = float(rec.get("estimated_saving_usd", 0.0))
                true = float(labels["true_saving_usd"])
                if true > 0 and abs(est - true) / true <= 0.2:
                    reward += 0.05
                    reason = "saving_estimate_accurate"

        if state["step"] > 20:
            penalty = 0.002 * (state["step"] - 20)
            reward -= penalty
            reason = "step_penalty"

        if state["query_count"] < 15 and action_type == "submit_report":
            reward += 0.05
            reason = "efficient_querying_bonus"

        self.cumulative += reward
        return reward, reason
