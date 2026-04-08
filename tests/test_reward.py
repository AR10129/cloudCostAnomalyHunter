import random

from env.environment import CloudCostEnv


def test_dense_reward_signal_random_episode() -> None:
    env = CloudCostEnv(task_name="task3_full_audit", seed=7)
    non_zero = 0
    total = 30

    resource_ids = [r["resource_id"] for r in env._state["infra_snapshot"]]
    for _ in range(total):
        action = random.choice(["query_billing", "query_infra", "write_note", "flag_anomaly"])
        if action == "query_billing":
            _, reward, _, _ = env.step({"action_type": "query_billing", "filter": {"service": "EC2"}})
        elif action == "query_infra":
            _, reward, _, _ = env.step({"action_type": "query_infra", "resource_id": random.choice(resource_ids)})
        elif action == "write_note":
            _, reward, _, _ = env.step({"action_type": "write_note", "content": "investigating spend drift"})
        else:
            _, reward, _, _ = env.step(
                {
                    "action_type": "flag_anomaly",
                    "resource_id": random.choice(resource_ids),
                    "anomaly_type": "zombie",
                    "severity": "medium",
                    "reasoning": "random probe",
                }
            )
        if reward != 0.0:
            non_zero += 1

    assert non_zero / total > 0.80
