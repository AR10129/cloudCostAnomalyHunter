from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class BillingSummary(BaseModel):
    total_cost_usd: float
    daily_avg_usd: float
    top_services: Dict[str, float]
    day_count: int


class ResourceInfo(BaseModel):
    resource_id: str
    service: str
    region: str
    owner: str
    launched_at: str
    last_accessed: str
    utilization_cpu_pct: float = 0.0
    inbound_traffic_mb: float = 0.0
    io_ops: int = 0
    daily_cost: float = 0.0


class AnomalyFlag(BaseModel):
    resource_id: str
    anomaly_type: str
    reasoning: str
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    root_cause_category: Optional[str] = None


class Recommendation(BaseModel):
    resource_id: str
    action_type: str
    estimated_saving_usd: float


class Observation(BaseModel):
    step: int
    billing_summary: BillingSummary
    recent_query_results: Optional[List[Dict[str, Any]]] = None
    infra_query_result: Optional[ResourceInfo] = None
    flagged_so_far: List[AnomalyFlag] = Field(default_factory=list)
    step_reward: float = 0.0
    done: bool = False


class StepReward(BaseModel):
    delta: float
    reason: str
    cumulative: float


class QueryBilling(BaseModel):
    action_type: Literal["query_billing"]
    filter: Dict[str, Any] = Field(default_factory=dict)


class QueryInfra(BaseModel):
    action_type: Literal["query_infra"]
    resource_id: str


class FlagAnomaly(BaseModel):
    action_type: Literal["flag_anomaly"]
    resource_id: str
    anomaly_type: str
    reasoning: str
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    root_cause_category: Optional[str] = None


class RecommendAction(BaseModel):
    action_type: Literal["recommend_action"]
    resource_id: str
    action_type_detail: str
    estimated_saving_usd: float


class WriteNote(BaseModel):
    action_type: Literal["write_note"]
    content: str


class SubmitReport(BaseModel):
    action_type: Literal["submit_report"]


Action = Union[QueryBilling, QueryInfra, FlagAnomaly, RecommendAction, WriteNote, SubmitReport]
