from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any, Dict, List


def _date_range(days: int, start: date) -> List[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(days)]


def _base_billing_rows(rng: random.Random, resources: List[Dict[str, Any]], days: int, start: date) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in _date_range(days, start):
        for res in resources:
            base = res.get("daily_cost", 8.0)
            jitter = rng.uniform(-0.2, 0.2) * base
            rows.append(
                {
                    "date": d,
                    "service": res["service"],
                    "resource_id": res["resource_id"],
                    "region": res["region"],
                    "cost_usd": round(max(0.1, base + jitter), 2),
                    "tags": {"team": res.get("owner", "platform")},
                }
            )
    return rows


def generate_task1(seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed)
    start = date(2025, 10, 1)
    resources = []
    zombies: List[str] = []
    for i in range(15):
        rid = f"ec2-{i:03d}"
        is_zombie = i in {2, 7, 13}
        if is_zombie:
            zombies.append(rid)
        resources.append(
            {
                "resource_id": rid,
                "service": "EC2",
                "region": rng.choice(["us-east-1", "us-west-2", "eu-west-1"]),
                "owner": rng.choice(["ml", "backend", "analytics"]),
                "launched_at": (start - timedelta(days=rng.randint(30, 150))).isoformat(),
                "last_accessed": (start - timedelta(days=rng.randint(35, 90))).isoformat() if is_zombie else (start - timedelta(days=rng.randint(1, 10))).isoformat(),
                "utilization_cpu_pct": 0.5 if is_zombie else round(rng.uniform(10.0, 70.0), 2),
                "inbound_traffic_mb": 0.0 if is_zombie else round(rng.uniform(20.0, 400.0), 2),
                "io_ops": 0 if is_zombie else rng.randint(100, 8000),
                "daily_cost": round(rng.uniform(6.0, 22.0), 2),
            }
        )

    billing = _base_billing_rows(rng, resources, days=90, start=start)
    return {
        "task": "task1_zombie",
        "billing": billing,
        "infra_snapshot": resources,
        "saas_licenses": [],
        "labels": {"zombies": zombies},
    }


def generate_task2(seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed + 1)
    start = date(2025, 1, 1)
    spike_variants = [
        ("EC2", "misconfigured_autoscale", "ec2-asg-17"),
        ("S3", "egress_surge", "s3-bucket-logs"),
        ("Lambda", "invocation_storm", "lambda-image-resizer"),
        ("RDS", "iops_burst", "rds-orders-primary"),
    ]
    service, cause, rid = spike_variants[rng.randint(0, len(spike_variants) - 1)]

    resources = [
        {"resource_id": rid, "service": service, "region": "us-east-1", "owner": "platform", "launched_at": "2024-01-01", "last_accessed": "2025-06-29", "utilization_cpu_pct": 55.0, "inbound_traffic_mb": 190.0, "io_ops": 2300, "daily_cost": 28.0},
        {"resource_id": "ec2-api-01", "service": "EC2", "region": "us-east-1", "owner": "backend", "launched_at": "2024-03-10", "last_accessed": "2025-06-29", "utilization_cpu_pct": 45.0, "inbound_traffic_mb": 420.0, "io_ops": 6400, "daily_cost": 19.0},
        {"resource_id": "s3-data-lake", "service": "S3", "region": "us-east-1", "owner": "analytics", "launched_at": "2023-09-02", "last_accessed": "2025-06-29", "utilization_cpu_pct": 0.0, "inbound_traffic_mb": 0.0, "io_ops": 14000, "daily_cost": 11.0},
    ]
    billing = _base_billing_rows(rng, resources, days=180, start=start)

    spike_start_idx = 142
    spike_end_idx = 150
    true_saving = 880.0
    dates = _date_range(180, start)
    for row in billing:
        if row["resource_id"] == rid and dates[spike_start_idx] <= row["date"] <= dates[spike_end_idx]:
            row["cost_usd"] = round(row["cost_usd"] * rng.uniform(3.2, 4.8), 2)

    return {
        "task": "task2_spike_rca",
        "billing": billing,
        "infra_snapshot": resources,
        "saas_licenses": [],
        "labels": {
            "spike_start_date": dates[spike_start_idx],
            "service": service,
            "resource_id": rid,
            "root_cause_category": cause,
            "true_saving_usd": true_saving,
            "valid_root_causes": ["misconfigured_autoscale", "invocation_storm", "egress_surge", "iops_burst"],
        },
    }


def generate_task3(seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed + 2)
    start = date(2025, 1, 1)

    resources = []
    for i in range(30):
        resources.append(
            {
                "resource_id": f"res-{i:03d}",
                "service": rng.choice(["EC2", "S3", "Lambda", "RDS", "BigQuery", "GKE"]),
                "region": rng.choice(["us-east-1", "us-west-2", "eu-west-1", "asia-south1"]),
                "owner": rng.choice(["ml", "backend", "analytics", "security"]),
                "launched_at": (start - timedelta(days=rng.randint(30, 300))).isoformat(),
                "last_accessed": (start + timedelta(days=rng.randint(220, 364))).isoformat(),
                "utilization_cpu_pct": round(rng.uniform(12.0, 80.0), 2),
                "inbound_traffic_mb": round(rng.uniform(10.0, 2000.0), 2),
                "io_ops": rng.randint(100, 12000),
                "daily_cost": round(rng.uniform(4.0, 38.0), 2),
            }
        )

    anomalies = {
        "zombie": {"resource_id": "res-003", "severity": "high"},
        "spike": {"resource_id": "res-011", "severity": "critical"},
        "overprovisioned_rds": {"resource_id": "res-019", "severity": "medium"},
        "unused_saas_seats": {"resource_id": "saas-zoom", "severity": "medium"},
        "cross_region_waste": {"resource_id": "res-025", "severity": "high"},
    }

    for r in resources:
        if r["resource_id"] == "res-003":
            r["utilization_cpu_pct"] = 0.3
            r["inbound_traffic_mb"] = 0.0
            r["io_ops"] = 0
            r["last_accessed"] = "2025-02-01"
            r["service"] = "EC2"
        if r["resource_id"] == "res-019":
            r["service"] = "RDS"
            r["utilization_cpu_pct"] = 4.2
            r["daily_cost"] = 46.0
        if r["resource_id"] == "res-011":
            r["service"] = "Lambda"
            r["daily_cost"] = 14.0
        if r["resource_id"] == "res-025":
            r["service"] = "S3"
            r["region"] = "us-east-1"
            r["daily_cost"] = 18.0

    billing = _base_billing_rows(rng, resources, days=365, start=start)
    for row in billing:
        if row["resource_id"] == "res-011" and "2025-10-10" <= row["date"] <= "2025-10-18":
            row["cost_usd"] = round(row["cost_usd"] * rng.uniform(3.5, 5.2), 2)
        if row["resource_id"] == "res-025":
            row["cost_usd"] = round(row["cost_usd"] * 1.25, 2)

    saas = [
        {"tool": "Zoom", "resource_id": "saas-zoom", "seats_purchased": 320, "seats_active": 190, "monthly_cost_usd": 6200},
        {"tool": "Notion", "resource_id": "saas-notion", "seats_purchased": 180, "seats_active": 175, "monthly_cost_usd": 2300},
        {"tool": "Datadog", "resource_id": "saas-datadog", "seats_purchased": 140, "seats_active": 126, "monthly_cost_usd": 4100},
    ]

    labels = {
        "anomalies": anomalies,
        "severity_weights": {"critical": 0.30, "high": 0.20, "medium": 0.10},
    }

    return {
        "task": "task3_full_audit",
        "billing": billing,
        "infra_snapshot": resources,
        "saas_licenses": saas,
        "labels": labels,
    }
