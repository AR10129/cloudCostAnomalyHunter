from __future__ import annotations

from datetime import date
from typing import Any, Dict


_EPS = 1e-3


def _within_days(a: str, b: str, days: int) -> bool:
    da = date.fromisoformat(a)
    db = date.fromisoformat(b)
    return abs((da - db).days) <= days


def grade(submission: Dict[str, Any], labels: Dict[str, Any]) -> float:
    score = 0.0
    report = submission.get("report", {})

    if _within_days(report.get("spike_start_date", "1900-01-01"), labels["spike_start_date"], 2):
        score += 0.25

    if report.get("service") == labels["service"]:
        score += 0.25

    if report.get("root_cause_category") == labels["root_cause_category"]:
        score += 0.25

    est = float(report.get("estimated_saving_usd", 0.0))
    true = float(labels["true_saving_usd"])
    if true > 0 and abs(est - true) / true <= 0.2:
        score += 0.25

    if score >= 1.0:
        score += 0.1

    return min(1.0 - _EPS, max(_EPS, score))
