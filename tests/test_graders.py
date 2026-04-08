from env.environment import CloudCostEnv


def test_task1_deterministic_score() -> None:
    env = CloudCostEnv(task_name="task1_zombie", seed=7)
    labels = env.state()["labels"]["zombies"]
    for rid in labels:
        env.step(
            {
                "action_type": "flag_anomaly",
                "resource_id": rid,
                "anomaly_type": "zombie",
                "severity": "high",
                "reasoning": "no utilization",
            }
        )
    _, _, _, info = env.step({"action_type": "submit_report"})
    assert 0.99 <= info["final_score"] < 1.0


def test_task2_deterministic_score() -> None:
    env = CloudCostEnv(task_name="task2_spike_rca", seed=7)
    labels = env.state()["labels"]
    env._state["report"].update(
        {
            "spike_start_date": labels["spike_start_date"],
            "service": labels["service"],
            "root_cause_category": labels["root_cause_category"],
            "estimated_saving_usd": labels["true_saving_usd"],
        }
    )
    _, _, _, info = env.step({"action_type": "submit_report"})
    assert 0.99 <= info["final_score"] < 1.0


def test_task3_deterministic_score() -> None:
    env = CloudCostEnv(task_name="task3_full_audit", seed=7)
    anomalies = env.state()["labels"]["anomalies"]
    for anomaly_type, payload in anomalies.items():
        env.step(
            {
                "action_type": "flag_anomaly",
                "resource_id": payload["resource_id"],
                "anomaly_type": anomaly_type,
                "severity": payload["severity"],
                "reasoning": "matched known signature",
            }
        )
    env._state["report"]["remediation_quality_bonus"] = 0.1
    _, _, _, info = env.step({"action_type": "submit_report"})
    assert 0.99 <= info["final_score"] < 1.0
