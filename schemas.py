"""Pydantic models for Hindsight.

Three groups:
  1. Request models          (what the frontend sends)
  2. AI output model         (what Gemini must return - validated leniently, so a
                              slightly messy answer is repaired instead of crashing)
  3. Response models         (what the frontend receives)
"""
import json
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# 1. Requests
# --------------------------------------------------------------------------- #
class IncidentCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200, examples=["Suspicious Login Activity"])
    description: Optional[str] = Field(
        "", max_length=20_000,
        examples=["Multiple failed logins followed by a successful login from an unusual IP."],
    )
    logs: Optional[str] = Field("", max_length=500_000, description="Raw logs / evidence pasted as text")

    @field_validator("title", "description", "logs", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        if v is None:
            return ""
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _require_some_evidence(self) -> "IncidentCreate":
        if not (self.description or "").strip() and not (self.logs or "").strip():
            raise ValueError("Provide a description and/or logs - the incident is empty")
        return self


# --------------------------------------------------------------------------- #
# 2. AI output (validated leniently on purpose)
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def _to_text(v: Any) -> str:
    """Turn whatever the model returned into a string."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "\n".join(_to_text(i) for i in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _to_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple)):
        return [_to_text(i) for i in v if i is not None and i != ""]
    return [_to_text(v)]


def _wrap_strings(v: Any, key: str) -> List[Any]:
    """If the model returned plain strings where objects were expected, wrap them."""
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    return [{key: i} if isinstance(i, str) else i for i in v]


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TimelineEvent(_Lenient):
    timestamp: str = ""
    event: str = ""
    importance: str = "MEDIUM"

    @field_validator("timestamp", "event", "importance", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return _to_text(v)

    @field_validator("importance")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper() or "MEDIUM"


class IndicatorOfCompromise(_Lenient):
    type: str = "other"
    value: str = ""
    reason: str = ""

    @field_validator("type", "value", "reason", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return _to_text(v)


class AffectedAsset(_Lenient):
    asset: str = ""
    impact: str = ""

    @field_validator("asset", "impact", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return _to_text(v)


class EvidenceItem(_Lenient):
    """One piece of reasoning, tagged by how strongly the data supports it."""
    detail: str = ""
    strength: str = "UNSPECIFIED"  # CONFIRMED | STRONG_INDICATION | HYPOTHESIS

    @field_validator("detail", "strength", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return _to_text(v)

    @field_validator("strength")
    @classmethod
    def _norm(cls, v: str) -> str:
        return v.strip().upper().replace(" ", "_") or "UNSPECIFIED"


class IncidentAnalysis(_Lenient):
    incident_type: str = "Unknown"
    severity: Severity
    confidence: float = Field(..., ge=0, le=1)
    executive_summary: str
    what_happened: str = ""
    why_suspicious: str = ""
    root_cause: str = ""
    timeline: List[TimelineEvent] = []
    indicators_of_compromise: List[IndicatorOfCompromise] = []
    affected_assets: List[AffectedAsset] = []
    attack_progression: List[str] = []
    possible_attack_techniques: List[str] = []
    impact_assessment: str = ""
    evidence: List[EvidenceItem] = []
    immediate_actions: List[str] = []
    containment_actions: List[str] = []
    investigation_actions: List[str] = []
    recovery_actions: List[str] = []
    preventive_actions: List[str] = []
    missing_information: List[str] = []

    # --- repair-before-validate helpers ---------------------------------- #
    @field_validator("severity", mode="before")
    @classmethod
    def _severity(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().upper()
            if v in ("INFO", "INFORMATIONAL"):
                return "LOW"
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                v = float(v.strip().rstrip("%"))
            except ValueError:
                return v
        if isinstance(v, (int, float)) and 1 < v <= 100:  # model answered 91 instead of 0.91
            return v / 100
        return v

    @field_validator(
        "incident_type", "executive_summary", "what_happened", "why_suspicious",
        "root_cause", "impact_assessment", mode="before",
    )
    @classmethod
    def _text(cls, v: Any) -> str:
        return _to_text(v)

    @field_validator(
        "attack_progression", "possible_attack_techniques", "immediate_actions",
        "containment_actions", "investigation_actions", "recovery_actions",
        "preventive_actions", "missing_information", mode="before",
    )
    @classmethod
    def _str_list(cls, v: Any) -> List[str]:
        return _to_str_list(v)

    @field_validator("timeline", mode="before")
    @classmethod
    def _timeline(cls, v: Any) -> List[Any]:
        return _wrap_strings(v, "event")

    @field_validator("indicators_of_compromise", mode="before")
    @classmethod
    def _iocs(cls, v: Any) -> List[Any]:
        return _wrap_strings(v, "value")

    @field_validator("affected_assets", mode="before")
    @classmethod
    def _assets(cls, v: Any) -> List[Any]:
        return _wrap_strings(v, "asset")

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v: Any) -> List[Any]:
        return _wrap_strings(v, "detail")


# --------------------------------------------------------------------------- #
# 3. Responses
# --------------------------------------------------------------------------- #
class IncidentCreated(BaseModel):
    incident_id: str
    status: str = "created"


class IncidentSummary(BaseModel):
    """One row of the incident list."""
    incident_id: str
    title: str
    incident_type: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    status: str
    created_at: str
    last_analyzed_at: Optional[str] = None


class IncidentDetail(IncidentSummary):
    description: str = ""
    raw_logs: str = ""
    analysis: Optional[IncidentAnalysis] = None
    analysis_source: Optional[str] = None  # "gemini" or "demo-mock"
    last_error: Optional[str] = None


class IncidentList(BaseModel):
    count: int
    incidents: List[IncidentSummary]


class Stats(BaseModel):
    total: int
    critical: int
    high: int
    medium: int
    low: int
    unanalyzed: int
    active_investigations: int
    recent: List[IncidentSummary]
