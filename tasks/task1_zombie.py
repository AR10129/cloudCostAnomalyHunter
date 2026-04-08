from __future__ import annotations

from typing import Any, Dict


def grade(submission: Dict[str, Any], labels: Dict[str, Any]) -> float:
    flagged = submission.get("flags", [])
    flagged_ids = {f.get("resource_id") for f in flagged}
    zombies = set(labels["zombies"])

    tp = len(flagged_ids & zombies)
    fp = len([fid for fid in flagged_ids if fid not in zombies])

    score = 0.0
    if tp == 3:
        score = 1.0
    elif tp == 2:
        score = 0.6
    elif tp == 1:
        score = 0.3

    score -= 0.1 * fp
    return max(0.0, min(1.0, score))
