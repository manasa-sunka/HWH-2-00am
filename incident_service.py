"""Business logic: validation -> storage -> AI analysis -> storage."""
import logging
import re
from typing import Any, Dict

from backend.database import database as db
from backend.models.schemas import IncidentCreate
from backend.services import ai_agent
from backend.services.ai_agent import AIAnalysisError

logger = logging.getLogger("hindsight.service")
_INCIDENT_ID = re.compile(r"^INC-\d{1,9}$")


class InvalidIncidentId(Exception):
    pass


class IncidentNotFound(Exception):
    pass


class EmptyIncident(Exception):
    pass


def normalize_id(incident_id: str) -> str:
    value = (incident_id or "").strip().upper()
    if not _INCIDENT_ID.match(value):
        raise InvalidIncidentId(f"'{incident_id}' is not a valid incident ID (expected format INC-001)")
    return value


def create_incident(payload: IncidentCreate) -> Dict[str, str]:
    return db.create_incident(payload.title, payload.description or "", payload.logs or "")


def list_incidents() -> Dict[str, Any]:
    items = db.list_incidents()
    return {"count": len(items), "incidents": items}


def get_incident(incident_id: str) -> Dict[str, Any]:
    incident = db.get_incident(normalize_id(incident_id))
    if incident is None:
        raise IncidentNotFound(f"Incident {incident_id.upper()} not found")
    return incident


def get_stats() -> Dict[str, Any]:
    return db.get_stats()


def analyze_incident(incident_id: str) -> Dict[str, Any]:
    incident = get_incident(incident_id)  # raises for bad/missing id
    if not (incident["description"] or "").strip() and not (incident["raw_logs"] or "").strip():
        raise EmptyIncident("This incident has no description or logs to analyze")

    try:
        analysis, source = ai_agent.analyze_incident(incident)
    except AIAnalysisError as exc:
        logger.warning("Analysis failed for %s: %s", incident["incident_id"], exc)
        try:
            db.mark_analysis_failed(incident["incident_id"], f"{exc.kind}: {exc.detail}")
        except db.DatabaseError:
            logger.exception("Could not record analysis failure")
        raise

    saved = db.save_analysis(incident["incident_id"], analysis.model_dump(mode="json"), source)
    return saved  # type: ignore[return-value]
