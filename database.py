"""SQLite storage for Hindsight (standard library only - no ORM).

Every function opens its own short-lived connection, which keeps things safe when
FastAPI runs endpoints in a thread pool. To move to PostgreSQL later, replace this
one file and keep the same function names.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "hindsight.db"

# Incident status values
STATUS_CREATED = "created"
STATUS_INVESTIGATING = "investigating"   # AI analysis done, analyst reviewing
STATUS_FAILED = "analysis_failed"


class DatabaseError(Exception):
    """Any storage failure. The API turns this into a clean 500 response."""


def _db_path() -> str:
    return os.getenv("DATABASE_PATH") or str(DEFAULT_DB_PATH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(_db_path(), timeout=10)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise DatabaseError(f"Cannot open database: {exc}") from exc
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise DatabaseError(str(exc)) from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                seq              INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id      TEXT UNIQUE,
                title            TEXT NOT NULL,
                description      TEXT NOT NULL DEFAULT '',
                raw_logs         TEXT NOT NULL DEFAULT '',
                created_at       TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'created',
                analysis         TEXT,
                analysis_source  TEXT,
                incident_type    TEXT,
                severity         TEXT,
                confidence       REAL,
                last_analyzed_at TEXT,
                last_error       TEXT
            )
            """
        )


def check_db() -> bool:
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except DatabaseError:
        return False


# --------------------------------------------------------------------------- #
def _row_to_dict(row: sqlite3.Row, full: bool) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "incident_id": row["incident_id"],
        "title": row["title"],
        "incident_type": row["incident_type"],
        "severity": row["severity"],
        "confidence": row["confidence"],
        "status": row["status"],
        "created_at": row["created_at"],
        "last_analyzed_at": row["last_analyzed_at"],
    }
    if full:
        data["description"] = row["description"]
        data["raw_logs"] = row["raw_logs"]
        data["analysis_source"] = row["analysis_source"]
        data["last_error"] = row["last_error"]
        try:
            data["analysis"] = json.loads(row["analysis"]) if row["analysis"] else None
        except json.JSONDecodeError:
            data["analysis"] = None
    return data


def create_incident(title: str, description: str, raw_logs: str) -> Dict[str, str]:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO incidents (title, description, raw_logs, created_at, status) VALUES (?, ?, ?, ?, ?)",
            (title, description, raw_logs, _now(), STATUS_CREATED),
        )
        incident_id = f"INC-{cur.lastrowid:03d}"
        conn.execute("UPDATE incidents SET incident_id = ? WHERE seq = ?", (incident_id, cur.lastrowid))
    return {"incident_id": incident_id, "status": STATUS_CREATED}


def get_incident(incident_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)).fetchone()
    return _row_to_dict(row, full=True) if row else None


def list_incidents() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM incidents ORDER BY seq DESC").fetchall()
    return [_row_to_dict(r, full=False) for r in rows]


def save_analysis(incident_id: str, analysis: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE incidents
               SET analysis = ?, analysis_source = ?, incident_type = ?, severity = ?,
                   confidence = ?, status = ?, last_analyzed_at = ?, last_error = NULL
             WHERE incident_id = ?
            """,
            (
                json.dumps(analysis, ensure_ascii=False),
                source,
                analysis.get("incident_type"),
                analysis.get("severity"),
                analysis.get("confidence"),
                STATUS_INVESTIGATING,
                _now(),
                incident_id,
            ),
        )
    return get_incident(incident_id)


def mark_analysis_failed(incident_id: str, error_text: str) -> None:
    """Record the failure. Keeps status 'investigating' if an earlier analysis exists."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE incidents
               SET last_error = ?,
                   status = CASE WHEN analysis IS NULL THEN ? ELSE status END
             WHERE incident_id = ?
            """,
            (error_text[:500], STATUS_FAILED, incident_id),
        )


def get_stats() -> Dict[str, Any]:
    incidents = list_incidents()
    by_sev = lambda s: sum(1 for i in incidents if i["severity"] == s)  # noqa: E731
    return {
        "total": len(incidents),
        "critical": by_sev("CRITICAL"),
        "high": by_sev("HIGH"),
        "medium": by_sev("MEDIUM"),
        "low": by_sev("LOW"),
        "unanalyzed": sum(1 for i in incidents if i["severity"] is None),
        "active_investigations": sum(1 for i in incidents if i["status"] == STATUS_INVESTIGATING),
        "recent": sorted(
            (i for i in incidents if i["last_analyzed_at"]),
            key=lambda i: i["last_analyzed_at"],
            reverse=True,
        )[:5],
    }
