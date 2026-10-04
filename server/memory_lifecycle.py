from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SERVER_TIMEZONE = os.getenv("MINDMATE_TIMEZONE", "Asia/Seoul")
PROMPT_TIME_BOUND_TYPES = {"task", "event"}
RECENT_COOKING_MEMORY_TYPES = {"recent_meal", "recent_cooking_event"}
RECENT_COOKING_RETENTION_DAYS = int(os.getenv("MINDMATE_RECENT_COOKING_RETENTION_DAYS", "7"))


def server_today() -> date:
    try:
        tz = ZoneInfo(SERVER_TIMEZONE)
    except ZoneInfoNotFoundError:
        return datetime.now().astimezone().date()
    return datetime.now(tz).date()


def is_past_time_bound_memory(
    *,
    memory_type: str | None,
    event_time: date | None,
    today: date | None = None,
) -> bool:
    if event_time is None:
        return False
    if (memory_type or "").strip().lower() not in PROMPT_TIME_BOUND_TYPES:
        return False
    return event_time < (today or server_today())


def derive_valid_until(
    *,
    memory_type: str | None,
    owner_character_id: str | None,
    event_time: date | None,
    explicit_valid_until: date | None,
) -> date | None:
    if explicit_valid_until is not None:
        return explicit_valid_until
    normalized_type = (memory_type or "").strip().lower()
    owner = (owner_character_id or "").strip().lower()
    if owner == "cooking" and normalized_type in RECENT_COOKING_MEMORY_TYPES and event_time is not None:
        return event_time + timedelta(days=RECENT_COOKING_RETENTION_DAYS)
    return None
