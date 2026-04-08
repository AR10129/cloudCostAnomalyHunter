from __future__ import annotations

from typing import Any, Dict


def grade(submission: Dict[str, Any], labels: Dict[str, Any]) -> float:
    flags = submission.get("flags", [])
    remediation_bonus = float(submission.get("report", {}).get("remediation_quality_bonus", 0.0))

    anomalies = labels["anomalies"]
    weights = labels["severity_weights"]
    by_resource = {f.get("resource_id"): f for f in flags}

    raw = 0.0
    false_positive = 0
    for anomaly_type, payload in anomalies.items():
        rid = payload["resource_id"]
        w = weights[payload["severity"]]
        if rid in by_resource:
            raw += 0.70 * w
            if by_resource[rid].get("anomaly_type") == anomaly_type:
                raw += 0.30 * w

    valid_ids = {payload["resource_id"] for payload in anomalies.values()}
    for f in flags:
        if f.get("resource_id") not in valid_ids:
            false_positive += 1

    raw -= min(0.25, 0.05 * false_positive)
    raw += min(0.10, max(0.0, remediation_bonus))

    return max(0.0, min(1.0, raw))
