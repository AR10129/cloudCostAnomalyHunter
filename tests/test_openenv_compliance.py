from env.environment import CloudCostEnv


def test_reset_returns_observation() -> None:
    env = CloudCostEnv(task_name="task1_zombie", seed=7)
    obs = env.reset()
    assert obs.step == 0
    assert obs.done is False


def test_step_signature_and_state_interface() -> None:
    env = CloudCostEnv(task_name="task1_zombie", seed=7)
    obs, reward, done, info = env.step({"action_type": "query_billing", "filter": {"service": "EC2"}})
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert obs.step == 1

    state = env.state()
    assert "task" in state
    assert "labels" in state


def test_submit_finishes_episode() -> None:
    env = CloudCostEnv(task_name="task1_zombie", seed=7)
    _, _, done, info = env.step({"action_type": "submit_report"})
    assert done is True
    assert "final_score" in info
