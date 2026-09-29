"""Input preprocessing and AI-output parsing helpers (no network, no database)."""
import json
import re
from typing import Any, Dict, List, Tuple

MAX_EVIDENCE_CHARS = 40_000   # keeps prompts fast and cheap for the hackathon MVP
MAX_ARTIFACTS_PER_TYPE = 25

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_URL = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
_HASH = re.compile(r"\b(?:[a-fA-F0-9]{64}|[a-fA-F0-9]{40}|[a-fA-F0-9]{32})\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


def clean_text(text: str) -> str:
    """Normalise newlines and drop control characters that can break prompts."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS.sub("", text).strip()


def truncate(text: str, limit: int = MAX_EVIDENCE_CHARS) -> Tuple[str, bool]:
    """Keep the start and end of very long logs (attacks are often at the tail)."""
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.6)
    tail = limit - head
    removed = len(text) - limit
    return f"{text[:head]}\n[... {removed} characters omitted ...]\n{text[-tail:]}", True


def _unique(items: List[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:MAX_ARTIFACTS_PER_TYPE]


def extract_artifacts(text: str) -> Dict[str, List[str]]:
    """Regex-extract candidate IOCs from the raw evidence (hints for the AI, not conclusions)."""
    return {
        "ip_addresses": _unique(_IPV4.findall(text)),
        "urls": _unique(_URL.findall(text)),
        "hashes": _unique(_HASH.findall(text)),
        "emails": _unique(_EMAIL.findall(text)),
    }


def build_evidence(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and package an incident's evidence for the AI prompt."""
    description = clean_text(incident.get("description", ""))
    logs, truncated = truncate(clean_text(incident.get("raw_logs", "")))
    combined = f"{description}\n{logs}"
    return {
        "title": clean_text(incident.get("title", "")),
        "description": description,
        "logs": logs,
        "logs_truncated": truncated,
        "log_line_count": logs.count("\n") + 1 if logs else 0,
        "artifacts": extract_artifacts(combined),
    }


def extract_json(raw: str) -> Dict[str, Any]:
    """Pull a JSON object out of a model reply, tolerating code fences and chatter.

    Raises ValueError if no valid JSON object can be found.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("AI returned an empty response")

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in AI response") from None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"AI response is not valid JSON: {exc}") from exc

    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError("AI response JSON is not an object")
    return obj
