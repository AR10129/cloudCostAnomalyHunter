---
title: CloudCostAnomalyHunter
sdk: docker
short_description: OpenEnv-style FinOps benchmark
tags:
  - openenv
  - finops
  - llm-eval
  - agent-benchmark
---

# Cloud Cost Anomaly Hunter

Cloud Cost Anomaly Hunter is an OpenEnv-style FinOps benchmark environment. It simulates NovaTech Inc., a multi-cloud company where an agent investigates cloud billing data, infrastructure metadata, and SaaS license usage to detect waste and recommend remediations.

## Environment Description

NovaTech runs workloads across AWS, GCP, and multiple SaaS tools. The agent acts as a FinOps analyst and must find hidden anomalies under step limits, then submit a final report scored by a deterministic grader.

## Observation Space

Observation is a typed structured payload represented by `Observation` in `env/models.py`.

- `step: int` current step index.
- `billing_summary: BillingSummary` aggregated cost statistics.
- `recent_query_results: Optional[List[Dict[str, Any]]]` latest billing query rows.
- `infra_query_result: Optional[ResourceInfo]` latest infra lookup payload.
- `flagged_so_far: List[AnomalyFlag]` anomaly flags submitted so far.
- `step_reward: float` dense incremental reward.
- `done: bool` episode completion flag.

## Action Space

The environment supports six function-call actions:

- `query_billing(filter: dict)` filters billing rows by exact field matches.
- `query_infra(resource_id: str)` retrieves resource metadata.
- `flag_anomaly(resource_id, anomaly_type, severity, reasoning, root_cause_category?)` logs anomaly detection.
- `recommend_action(resource_id, action_type_detail, estimated_saving_usd)` logs remediation recommendation.
- `write_note(content: str)` appends analyst notes to scratchpad.
- `submit_report()` finalizes the episode and triggers grading.

## Reward Function

Dense reward is implemented in `env/reward.py`.

- True positive flag: `+0.15 to +0.30` depending on task severity weighting.
- False positive flag: `-0.05` each.
- Root cause match (Task 2): `+0.10`.
- Saving estimate within 20% (Task 2): `+0.05`.
- Efficient querying (<15 queries at submit): `+0.05`.
- Step penalty after step 20: `-0.002` per extra step.

## Task Descriptions

### Task 1: Zombie Resource Detection (Easy)

Input: 90-day billing + infra snapshot. Objective: find 3 injected zombie resources.

Expected baseline range:

- GPT-4o: `0.70 - 0.85`

### Task 2: Cost Spike Root-Cause Analysis (Medium)

Input: 180-day billing with one injected spike. Objective: detect spike date, service, root cause, and saving estimate.

Expected baseline range:

- GPT-4o: `0.55 - 0.70`

### Task 3: Multi-Vector Anomaly Audit (Hard)

Input: 365-day billing + infra snapshot + SaaS licenses. Objective: identify and classify five anomaly types, then provide remediation quality.

Expected baseline range:

- GPT-4o: `0.40 - 0.55`

## Setup Instructions

### Local Python

```bash
pip install -r requirements.txt
```

### Docker Build

```bash
docker build -t cloud-cost-anomaly-hunter .
```

### Docker Run

```bash
docker run --rm -p 7860:7860 cloud-cost-anomaly-hunter
```

## Running the Inference Script

Submission entrypoint is `inference.py` at repo root.
This is the OpenEnv submission script expected by validators.

Required variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```bash
export API_BASE_URL=https://your-openai-compatible-endpoint/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=your_token
python inference.py
```

The script emits structured stdout logs using `[START]`, `[STEP]`, and `[END]` records.

## Running the Baseline

`baseline/run_baseline.py` is a local heuristic smoke script for reproducibility checks.
It is not the official OpenEnv submission entrypoint.

```bash
export OPENAI_API_KEY=your_key_here
python baseline/run_baseline.py
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_key_here"
python baseline/run_baseline.py
```

## Baseline Scores

| Task | Expected GPT-4o | Perfect |
|---|---|---|
| Task 1: Zombie Detection | 0.70 - 0.85 | 1.00 |
| Task 2: Spike RCA | 0.55 - 0.70 | 1.00 |
| Task 3: Full Audit | 0.40 - 0.55 | 1.00 |
| Aggregate mean | 0.55 - 0.70 | 1.00 |

## Contributing / Extending

- Add anomaly templates in `data/anomaly_templates.json`.
- Extend synthetic data logic in `env/data_generator.py`.
- Add a new grader in `tasks/` and wire it in `env/environment.py`.
- Add tests in `tests/` for determinism, reward density, and compliance.

## Pre-Submission Validator

Run:

```bash
python scripts/pre_submission_validate.py
```

This checks core submission prerequisites (metadata, inference entrypoint, tasks/graders, and Space endpoints).
