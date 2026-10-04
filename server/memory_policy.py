from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


def apply_memory_policy(
    conn: Any,
    *,
    user_id: str,
    active_character_id: str,
    memories: list[dict[str, Any]],
    character: dict[str, Any] | None = None,
    query: str | None = None,
    log_decisions: bool = False,
    allow_sensitive: bool = False,
) -> list[dict[str, Any]]:
    """Apply deterministic MAP rules to candidate memories.

    The policy is deliberately simple for the MVP:
    - shared memories are allowed
    - active character private memories are allowed
    - other character private memories are blocked
    - blocked/outdated/sensitive memories are excluded from prompt packs
    """
    allowed_domains = set(character.get("allowed_domains") or []) if character else set()
    blocked_domains = set(character.get("blocked_domains") or []) if character else set()
    decisions: list[dict[str, Any]] = []

    for memory in memories:
        decision = decide_memory(
            memory,
            active_character_id=active_character_id,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            allow_sensitive=allow_sensitive,
        )
        if log_decisions:
            log_policy_decision(
                conn,
                user_id=user_id,
                active_character_id=active_character_id,
                query=query,
                memory_id=memory["memory_id"],
                decision=decision["decision"],
                reason=decision["reason"],
                score=decision["score"],
            )
        if decision["decision"] == "allow":
            item = dict(memory)
            item["policy_score"] = decision["score"]
            item["policy_reason"] = decision["reason"]
            item["_rank_score"] = rank_score(item)
            decisions.append(item)

    decisions.sort(
        key=lambda item: (
            item.get("_rank_score") or 0.0,
            item.get("policy_score") or 0.0,
            item.get("event_time") or date.min,
            item.get("memory_id") or 0,
        ),
        reverse=True,
    )
    return decisions


def rank_score(memory: dict[str, Any]) -> float:
    """Use retrieval relevance when present; otherwise keep policy-only ordering."""
    policy_score = float(memory.get("policy_score") or 0.0)
    if memory.get("score") is None:
        return policy_score
    return float(memory.get("score") or 0.0) + (0.25 * policy_score)


def decide_memory(
    memory: dict[str, Any],
    *,
    active_character_id: str,
    allowed_domains: set[str],
    blocked_domains: set[str],
    allow_sensitive: bool = False,
) -> dict[str, Any]:
    scope = memory.get("scope") or "shared"
    owner = memory.get("owner_character_id")
    tags = set(memory.get("domain_tags") or [])
    status = memory.get("validity_status") or "current"
    sensitivity = memory.get("sensitivity") or "normal"

    if scope == "blocked":
        return deny("scope=blocked")
    if status == "outdated":
        return deny("validity_status=outdated")
    if sensitivity == "sensitive" and not allow_sensitive:
        return deny("sensitivity=sensitive")
    if scope == "character_private" and owner != active_character_id:
        return deny(f"private memory belongs to {owner or 'unknown'}")
    if tags & blocked_domains:
        return deny("domain tag is blocked for active character")

    score = base_score(memory)
    reasons = ["allowed"]
    if scope == "character_private":
        score += 0.35
        reasons.append("active-character private")
    if tags & allowed_domains:
        score += 0.15
        reasons.append("domain match")
    if status == "uncertain":
        score -= 0.25
        reasons.append("uncertain")

    return {
        "decision": "allow",
        "reason": ", ".join(reasons),
        "score": max(0.0, score),
    }


def base_score(memory: dict[str, Any]) -> float:
    importance = float(memory.get("importance") or 3) / 5.0
    confidence = float(memory.get("confidence") or 1.0)
    recency = recency_score(memory)
    return (0.45 * importance) + (0.35 * confidence) + (0.20 * recency)


def recency_score(memory: dict[str, Any]) -> float:
    event_time = memory.get("event_time")
    created_at = memory.get("created_at")
    if isinstance(event_time, date):
        age_days = (date.today() - event_time).days
    elif isinstance(created_at, datetime):
        now = datetime.now(timezone.utc)
        ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
        age_days = int((now - ts).total_seconds() / 86400)
    else:
        age_days = 0
    return max(0.0, min(1.0, 1.0 / (1.0 + max(age_days, 0) / 30.0)))


def deny(reason: str) -> dict[str, Any]:
    return {"decision": "block", "reason": reason, "score": 0.0}


def log_policy_decision(
    conn: Any,
    *,
    user_id: str,
    active_character_id: str,
    query: str | None,
    memory_id: int,
    decision: str,
    reason: str,
    score: float,
) -> None:
    conn.execute(
        """
        INSERT INTO policy_decisions (
            user_id, active_character_id, query, memory_id, decision, reason, score
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (user_id, active_character_id, query, memory_id, decision, reason, score),
    )
