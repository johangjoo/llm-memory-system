from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from server.memory_policy import apply_memory_policy
from server.memory_lifecycle import PROMPT_TIME_BOUND_TYPES, server_today


SHARED_SECTION = "shared"
CHARACTER_SECTIONS = {"study", "cooking"}
SECTION_CANDIDATE_STATUSES = ("candidate", "selected")
PARK_AFTER_MISSES = 3
PARK_IMPORTANCE_MAX = 3
DEFAULT_MEMORY_LIMIT = 0
DEFAULT_REFLECTION_LIMIT = 0
DEFAULT_MAX_PROMPT_CHARS = 12000
MAX_UNBOUNDED_REFLECTIONS = 50


def build_character_memory_pack(
    conn: Any,
    *,
    user_id: str,
    character_id: str,
    shared_limit: int = DEFAULT_MEMORY_LIMIT,
    private_limit: int = DEFAULT_MEMORY_LIMIT,
    reflection_limit: int = DEFAULT_REFLECTION_LIMIT,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    log_decisions: bool = False,
) -> dict[str, Any]:
    """Build a Realtime memory pack directly from facts without using cache."""
    user = require_user(conn, user_id)
    character = require_character(conn, character_id)
    shared_section = build_memory_section(
        conn,
        user_id=user_id,
        section_key=SHARED_SECTION,
        memory_limit=shared_limit,
        max_section_chars=section_char_limit(max_prompt_chars),
        log_decisions=log_decisions,
    )
    private_section = build_memory_section(
        conn,
        user_id=user_id,
        section_key=section_key_for_character(character_id),
        memory_limit=private_limit,
        max_section_chars=section_char_limit(max_prompt_chars),
        log_decisions=log_decisions,
    )
    reflections = fetch_reflections(
        conn,
        user_id=user_id,
        character_id=character_id,
        reflection_limit=reflection_limit,
    )
    return compose_character_memory_pack(
        user=user,
        character=character,
        shared_section=shared_section,
        private_section=private_section,
        reflections=reflections,
        max_prompt_chars=max_prompt_chars,
    )


def get_or_build_character_memory_pack(
    conn: Any,
    *,
    user_id: str,
    character_id: str,
    shared_limit: int = DEFAULT_MEMORY_LIMIT,
    private_limit: int = DEFAULT_MEMORY_LIMIT,
    reflection_limit: int = DEFAULT_REFLECTION_LIMIT,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    refresh_cache: bool = False,
    log_decisions: bool = False,
) -> dict[str, Any]:
    """Read cached memory sections, then assemble the active character prompt."""
    user = require_user(conn, user_id)
    character = require_character(conn, character_id)
    max_section_chars = section_char_limit(max_prompt_chars)
    shared_section = get_or_build_memory_section(
        conn,
        user_id=user_id,
        section_key=SHARED_SECTION,
        memory_limit=shared_limit,
        max_section_chars=max_section_chars,
        refresh_cache=refresh_cache,
        log_decisions=log_decisions,
    )
    private_section = get_or_build_memory_section(
        conn,
        user_id=user_id,
        section_key=section_key_for_character(character_id),
        memory_limit=private_limit,
        max_section_chars=max_section_chars,
        refresh_cache=refresh_cache,
        log_decisions=log_decisions,
    )
    reflections = fetch_reflections(
        conn,
        user_id=user_id,
        character_id=character_id,
        reflection_limit=reflection_limit,
    )
    return compose_character_memory_pack(
        user=user,
        character=character,
        shared_section=shared_section,
        private_section=private_section,
        reflections=reflections,
        max_prompt_chars=max_prompt_chars,
    )


def build_memory_section(
    conn: Any,
    *,
    user_id: str,
    section_key: str,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    max_section_chars: int = 2500,
    log_decisions: bool = False,
) -> dict[str, Any]:
    """Select the compact, prompt-ready memories for one user memory section."""
    section_key = normalize_section_key(section_key)
    character = None if section_key == SHARED_SECTION else require_character(conn, section_key)
    candidates = fetch_section_candidates(conn, user_id=user_id, section_key=section_key)
    allowed = apply_memory_policy(
        conn,
        user_id=user_id,
        active_character_id=section_key,
        memories=candidates,
        character=character,
        query=f"memory_section:{section_key}",
        log_decisions=log_decisions,
    )
    selected = select_memories_for_section(
        section_key=section_key,
        memories=allowed,
        memory_limit=memory_limit,
        max_section_chars=max_section_chars,
    )
    included_ids = [row["memory_id"] for row in selected]
    allowed_ids = {row["memory_id"] for row in allowed}
    blocked_ids = [row["memory_id"] for row in candidates if row["memory_id"] not in allowed_ids]
    section_text = render_memory_section(
        section_key=section_key,
        memories=selected,
        max_section_chars=max_section_chars,
    )
    counts = {
        "candidates": len(candidates),
        "allowed": len(allowed),
        "included": len(selected),
        "blocked": len(blocked_ids),
    }
    return {
        "section_key": section_key,
        "section_text": section_text,
        "included_memory_ids": included_ids,
        "blocked_memory_ids": blocked_ids,
        "allowed_memory_ids": [row["memory_id"] for row in allowed],
        "selected_memories": selected,
        "counts": counts,
        "memory_limit": memory_limit,
        "max_section_chars": max_section_chars,
    }


def get_or_build_memory_section(
    conn: Any,
    *,
    user_id: str,
    section_key: str,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    max_section_chars: int = 2500,
    refresh_cache: bool = False,
    updated_from_session_id: str | None = None,
    log_decisions: bool = False,
) -> dict[str, Any]:
    if not refresh_cache:
        cached = read_memory_section_cache(
            conn,
            user_id=user_id,
            section_key=section_key,
            memory_limit=memory_limit,
            max_section_chars=max_section_chars,
        )
        if cached is not None:
            return cached

    return rebuild_memory_section_cache(
        conn,
        user_id=user_id,
        section_key=section_key,
        memory_limit=memory_limit,
        max_section_chars=max_section_chars,
        updated_from_session_id=updated_from_session_id,
        log_decisions=log_decisions,
    )


def rebuild_memory_section_cache(
    conn: Any,
    *,
    user_id: str,
    section_key: str,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    max_section_chars: int = 2500,
    updated_from_session_id: str | None = None,
    log_decisions: bool = False,
) -> dict[str, Any]:
    archive_expired_prompt_memories(conn, user_id=user_id, section_key=section_key)
    park_past_time_bound_prompt_memories(conn, user_id=user_id, section_key=section_key)
    section = build_memory_section(
        conn,
        user_id=user_id,
        section_key=section_key,
        memory_limit=memory_limit,
        max_section_chars=max_section_chars,
        log_decisions=log_decisions,
    )
    update_memory_selection_statuses(
        conn,
        user_id=user_id,
        section_key=section["section_key"],
        selected_memories=section.get("selected_memories", []),
        allowed_memory_ids=section.get("allowed_memory_ids", []),
    )
    conn.execute(
        """
        INSERT INTO user_memory_sections (
            user_id, section_key, section_text, included_memory_ids,
            blocked_memory_ids, counts, memory_limit, max_section_chars,
            build_mode, updated_from_session_id, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id, section_key) DO UPDATE SET
            section_text = EXCLUDED.section_text,
            included_memory_ids = EXCLUDED.included_memory_ids,
            blocked_memory_ids = EXCLUDED.blocked_memory_ids,
            counts = EXCLUDED.counts,
            memory_limit = EXCLUDED.memory_limit,
            max_section_chars = EXCLUDED.max_section_chars,
            build_mode = EXCLUDED.build_mode,
            updated_from_session_id = EXCLUDED.updated_from_session_id,
            updated_at = NOW()
        """,
        (
            user_id,
            section["section_key"],
            section["section_text"],
            section["included_memory_ids"],
            section["blocked_memory_ids"],
            Jsonb(section["counts"]),
            memory_limit,
            max_section_chars,
            "score_pruned_sections",
            updated_from_session_id,
        ),
    )
    section["cache"] = {"hit": False, "updated": True}
    return section


def read_memory_section_cache(
    conn: Any,
    *,
    user_id: str,
    section_key: str,
    memory_limit: int,
    max_section_chars: int,
) -> dict[str, Any] | None:
    section_key = normalize_section_key(section_key)
    row = conn.execute(
        """
        SELECT section_key, section_text, included_memory_ids, blocked_memory_ids,
               counts, memory_limit, max_section_chars, updated_at
        FROM user_memory_sections
        WHERE user_id = %s
          AND section_key = %s
          AND memory_limit = %s
          AND max_section_chars = %s
        """,
        (user_id, section_key, memory_limit, max_section_chars),
    ).fetchone()
    if row is None:
        return None
    return {
        "section_key": row["section_key"],
        "section_text": row["section_text"] or "",
        "included_memory_ids": list(row["included_memory_ids"] or []),
        "blocked_memory_ids": list(row["blocked_memory_ids"] or []),
        "counts": dict(row["counts"] or {}),
        "memory_limit": row["memory_limit"],
        "max_section_chars": row["max_section_chars"],
        "cache": {"hit": True, "updated_at": row["updated_at"]},
    }


def refresh_memory_sections_after_fact_update(
    conn: Any,
    *,
    user_id: str,
    results: list[dict[str, Any]],
    active_character_id: str | None = None,
    session_id: str | None = None,
) -> list[str]:
    section_keys = affected_memory_sections(
        conn,
        results=results,
        active_character_id=active_character_id,
    )
    for section_key in section_keys:
        rebuild_memory_section_cache(
            conn,
            user_id=user_id,
            section_key=section_key,
            updated_from_session_id=session_id,
        )
    return section_keys


def rebuild_all_memory_sections(
    conn: Any,
    *,
    user_id: str,
    session_id: str | None = None,
) -> list[str]:
    section_keys = [SHARED_SECTION, *list_character_ids(conn)]
    for section_key in section_keys:
        rebuild_memory_section_cache(
            conn,
            user_id=user_id,
            section_key=section_key,
            updated_from_session_id=session_id,
        )
    return section_keys


def affected_memory_sections(
    conn: Any,
    *,
    results: list[dict[str, Any]],
    active_character_id: str | None,
) -> list[str]:
    affected: set[str] = set()
    expired_ids: list[int] = []
    for result in results:
        fact = result.get("fact")
        if fact:
            affected.add(section_key_for_fact(fact, active_character_id=active_character_id))
        if result.get("expired_id"):
            expired_ids.append(int(result["expired_id"]))

    if expired_ids:
        rows = conn.execute(
            """
            SELECT scope, owner_character_id
            FROM memory_facts
            WHERE memory_id = ANY(%s)
            """,
            (expired_ids,),
        ).fetchall()
        for row in rows:
            affected.add(section_key_for_fact(row, active_character_id=active_character_id))

    return sorted(key for key in affected if key)


def section_key_for_fact(
    fact: dict[str, Any],
    *,
    active_character_id: str | None = None,
) -> str:
    scope = fact.get("scope") or SHARED_SECTION
    owner = fact.get("owner_character_id")
    if scope == "character_private":
        return section_key_for_character(owner or active_character_id or "")
    return SHARED_SECTION


def section_key_for_character(character_id: str) -> str:
    if not character_id:
        return SHARED_SECTION
    return normalize_section_key(character_id)


def normalize_section_key(section_key: str) -> str:
    key = (section_key or SHARED_SECTION).strip().lower()
    if key == SHARED_SECTION:
        return key
    if key in CHARACTER_SECTIONS:
        return key
    raise CharacterNotFound(key)


def fetch_section_candidates(
    conn: Any,
    *,
    user_id: str,
    section_key: str,
) -> list[dict[str, Any]]:
    if section_key == SHARED_SECTION:
        where = "user_id = %s AND scope = 'shared'"
        params: tuple[Any, ...] = (user_id,)
    else:
        where = "user_id = %s AND scope = 'character_private' AND owner_character_id = %s"
        params = (user_id, section_key)

    today = server_today()
    query_params = (*params, today, list(PROMPT_TIME_BOUND_TYPES), today)

    return conn.execute(
        f"""
        SELECT memory_id, external_id, user_id, content, summary_for_prompt,
               memory_type, scope, owner_character_id, domain_tags, importance,
               event_time, valid_until, validity_status, confidence, sensitivity,
               conflict_group_id, selection_status, selection_score,
               selected_count, selection_miss_count, last_selected_at,
               last_reviewed_at, parked_reason, source_session_id, created_at
        FROM memory_facts
        WHERE {where}
          AND (valid_until IS NULL OR valid_until > %s)
          AND validity_status <> 'outdated'
          AND sensitivity <> 'sensitive'
          AND selection_status IN ('candidate', 'selected')
          AND NOT (
              memory_type = ANY(%s)
              AND event_time IS NOT NULL
              AND event_time < %s
          )
        ORDER BY COALESCE(event_time, created_at::date) DESC NULLS LAST,
                 importance DESC NULLS LAST,
                 memory_id DESC
        """,
        query_params,
    ).fetchall()


def archive_expired_prompt_memories(
    conn: Any,
    *,
    user_id: str,
    section_key: str,
) -> None:
    if section_key == SHARED_SECTION:
        where = "user_id = %s AND scope = 'shared'"
        params: tuple[Any, ...] = (user_id,)
    else:
        where = "user_id = %s AND scope = 'character_private' AND owner_character_id = %s"
        params = (user_id, section_key)

    today = server_today()
    conn.execute(
        f"""
        UPDATE memory_facts
        SET selection_status = 'archived',
            last_reviewed_at = NOW(),
            parked_reason = %s
        WHERE {where}
          AND selection_status IN ('candidate', 'selected', 'parked')
          AND valid_until IS NOT NULL
          AND valid_until <= %s
        """,
        (
            "valid_until elapsed; archived from prompt selection",
            *params,
            today,
        ),
    )


def park_past_time_bound_prompt_memories(
    conn: Any,
    *,
    user_id: str,
    section_key: str,
) -> None:
    if section_key == SHARED_SECTION:
        where = "user_id = %s AND scope = 'shared'"
        params: tuple[Any, ...] = (user_id,)
    else:
        where = "user_id = %s AND scope = 'character_private' AND owner_character_id = %s"
        params = (user_id, section_key)

    today = server_today()
    conn.execute(
        f"""
        UPDATE memory_facts
        SET selection_status = 'parked',
            last_reviewed_at = NOW(),
            parked_reason = %s
        WHERE {where}
          AND selection_status IN ('candidate', 'selected')
          AND memory_type = ANY(%s)
          AND event_time IS NOT NULL
          AND event_time < %s
          AND (valid_until IS NULL OR valid_until > %s)
          AND validity_status <> 'outdated'
          AND sensitivity <> 'sensitive'
        """,
        (
            "past task/event excluded from prompt selection by server date",
            *params,
            list(PROMPT_TIME_BOUND_TYPES),
            today,
            today,
        ),
    )


def update_memory_selection_statuses(
    conn: Any,
    *,
    user_id: str,
    section_key: str,
    selected_memories: list[dict[str, Any]],
    allowed_memory_ids: list[int],
) -> None:
    selected_ids = [int(row["memory_id"]) for row in selected_memories]
    allowed_ids = [int(memory_id) for memory_id in allowed_memory_ids]
    missed_ids = [memory_id for memory_id in allowed_ids if memory_id not in set(selected_ids)]

    for memory in selected_memories:
        conn.execute(
            """
            UPDATE memory_facts
            SET selection_status = 'selected',
                selection_score = %s,
                selected_count = selected_count + 1,
                selection_miss_count = 0,
                last_selected_at = NOW(),
                last_reviewed_at = NOW(),
                parked_reason = NULL
            WHERE user_id = %s AND memory_id = %s
            """,
            (
                float(memory.get("policy_score") or 0.0),
                user_id,
                memory["memory_id"],
            ),
        )

    if missed_ids:
        conn.execute(
            """
            UPDATE memory_facts
            SET selection_miss_count = selection_miss_count + 1,
                last_reviewed_at = NOW(),
                selection_status = CASE
                    WHEN selection_miss_count + 1 >= %s
                         AND COALESCE(importance, 3) <= %s
                    THEN 'parked'
                    ELSE 'candidate'
                END,
                parked_reason = CASE
                    WHEN selection_miss_count + 1 >= %s
                         AND COALESCE(importance, 3) <= %s
                    THEN %s
                    ELSE parked_reason
                END
            WHERE user_id = %s
              AND memory_id = ANY(%s)
              AND selection_status IN ('candidate', 'selected')
            """,
            (
                PARK_AFTER_MISSES,
                PARK_IMPORTANCE_MAX,
                PARK_AFTER_MISSES,
                PARK_IMPORTANCE_MAX,
                f"missed {PARK_AFTER_MISSES} {section_key} section selections",
                user_id,
                missed_ids,
            ),
        )


def fetch_reflections(
    conn: Any,
    *,
    user_id: str,
    character_id: str,
    reflection_limit: int,
) -> list[dict[str, Any]]:
    sql_limit = MAX_UNBOUNDED_REFLECTIONS if reflection_limit <= 0 else reflection_limit
    return conn.execute(
        """
        SELECT reflection_id, character_id, summary, source_memory_ids, created_at
        FROM reflections
        WHERE user_id = %s AND (character_id IS NULL OR character_id = %s)
        ORDER BY created_at DESC, reflection_id DESC
        LIMIT %s
        """,
        (user_id, character_id, sql_limit),
    ).fetchall()


def compose_character_memory_pack(
    *,
    user: dict[str, Any],
    character: dict[str, Any],
    shared_section: dict[str, Any],
    private_section: dict[str, Any],
    reflections: list[dict[str, Any]],
    max_prompt_chars: int,
) -> dict[str, Any]:
    text = render_pack_from_sections(
        user=user,
        character=character,
        shared_section_text=shared_section.get("section_text") or "",
        private_section_text=private_section.get("section_text") or "",
        reflections=reflections,
        max_prompt_chars=max_prompt_chars,
    )
    counts = {
        "candidates": int(shared_section["counts"].get("candidates", 0))
        + int(private_section["counts"].get("candidates", 0)),
        "included": int(shared_section["counts"].get("included", 0))
        + int(private_section["counts"].get("included", 0)),
        "blocked": int(shared_section["counts"].get("blocked", 0))
        + int(private_section["counts"].get("blocked", 0)),
        "shared": int(shared_section["counts"].get("included", 0)),
        "character_private": int(private_section["counts"].get("included", 0)),
        "reflections": len(reflections),
    }
    return {
        "text": text,
        "user": dict(user),
        "character": dict(character),
        "counts": counts,
        "included_memory_ids": [
            *shared_section.get("included_memory_ids", []),
            *private_section.get("included_memory_ids", []),
        ],
        "blocked_memory_ids": [
            *shared_section.get("blocked_memory_ids", []),
            *private_section.get("blocked_memory_ids", []),
        ],
        "cache": {
            "sections": {
                shared_section["section_key"]: shared_section.get("cache"),
                private_section["section_key"]: private_section.get("cache"),
            }
        },
    }


def render_pack_from_sections(
    *,
    user: dict[str, Any],
    character: dict[str, Any],
    shared_section_text: str,
    private_section_text: str,
    reflections: list[dict[str, Any]],
    max_prompt_chars: int,
) -> str:
    lines: list[str] = [
        "## Active Character",
        f"- id: {character['character_id']}",
        f"- name: {character['name']}",
    ]
    if character.get("description"):
        lines.append(f"- role: {character['description']}")
    if character.get("tone"):
        lines.append(f"- tone: {character['tone']}")
    if character.get("system_prompt"):
        lines.extend(["", "## Character Prompt", str(character["system_prompt"]).strip()])
    if character.get("switch_rules"):
        lines.extend(["", "## Character Switch Rules", str(character["switch_rules"]).strip()])

    lines.extend(
        [
            "",
            "## Memory Use Rules",
            "- Use the user profile, shared memory, and this character's memory when relevant.",
            "- Do not use private memory owned by another character.",
            "- If the current character is not suitable, call switch_character.",
            "- Do not mention memory IDs or database details to the user.",
            "",
            "## User Profile",
        ]
    )
    for label, key in (
        ("name", "user_name"),
        ("robot_name", "robot_name"),
        ("age", "age"),
        ("job", "job"),
        ("location", "location"),
        ("living", "living_info"),
        ("habit", "habit"),
        ("persona", "persona"),
    ):
        if user.get(key):
            lines.append(f"- {label}: {user[key]}")

    if shared_section_text.strip():
        lines.extend(["", shared_section_text.strip()])
    if private_section_text.strip():
        lines.extend(["", private_section_text.strip()])

    if reflections:
        lines.extend(["", "## Current Insights"])
        for reflection in reflections:
            lines.append(f"- {reflection['summary']}")

    text = "\n".join(lines).strip() + "\n"
    if len(text) <= max_prompt_chars:
        return text
    return text[: max_prompt_chars - 32].rstrip() + "\n...[trimmed]\n"


def render_memory_section(
    *,
    section_key: str,
    memories: list[dict[str, Any]],
    max_section_chars: int,
) -> str:
    if not memories:
        return ""
    lines = [section_title(section_key)]
    for memory in memories:
        next_line = format_memory_line(memory)
        candidate_text = "\n".join([*lines, next_line]).strip() + "\n"
        if len(candidate_text) > max_section_chars and len(lines) > 1:
            break
        lines.append(next_line)

    text = "\n".join(lines).strip() + "\n"
    if len(text) <= max_section_chars:
        return text
    return text[: max_section_chars - 24].rstrip() + "\n...[trimmed]\n"


def select_memories_for_section(
    *,
    section_key: str,
    memories: list[dict[str, Any]],
    memory_limit: int,
    max_section_chars: int,
) -> list[dict[str, Any]]:
    """Select by score order until the section character budget is full.

    memory_limit > 0 keeps the old hard-count behavior for explicit API callers.
    memory_limit <= 0 means "no count cap"; the section is bounded by chars only.
    """
    selected: list[dict[str, Any]] = []
    lines = [section_title(section_key)]
    source = memories[:memory_limit] if memory_limit > 0 else memories
    for memory in source:
        next_line = format_memory_line(memory)
        candidate_text = "\n".join([*lines, next_line]).strip() + "\n"
        if len(candidate_text) > max_section_chars and selected:
            break
        selected.append(memory)
        lines.append(next_line)
        if len(candidate_text) > max_section_chars:
            break
    return selected


def format_memory_line(memory: dict[str, Any]) -> str:
    summary = memory.get("summary_for_prompt") or memory["content"]
    memory_type = f"[{memory['memory_type']}] " if memory.get("memory_type") else ""
    event = f" ({memory['event_time']})" if memory.get("event_time") else ""
    return f"- {memory_type}{summary}{event}"


def section_title(section_key: str) -> str:
    if section_key == SHARED_SECTION:
        return "## Shared Memory"
    return f"## {section_key.capitalize()} Memory"


def section_char_limit(max_prompt_chars: int) -> int:
    return max(1200, min(5200, int(max_prompt_chars * 0.38)))


def require_user(conn: Any, user_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT user_id, user_name, age, job, location, living_info, habit, persona, robot_name
        FROM users WHERE user_id = %s
        """,
        (user_id,),
    ).fetchone()
    if row is None:
        raise UserNotFound(user_id)
    return row


def require_character(conn: Any, character_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT character_id, name, description, system_prompt, switch_rules,
               allowed_domains, blocked_domains, tone
        FROM characters WHERE character_id = %s
        """,
        (character_id,),
    ).fetchone()
    if row is None:
        raise CharacterNotFound(character_id)
    return row


def list_character_ids(conn: Any) -> list[str]:
    rows = conn.execute(
        "SELECT character_id FROM characters ORDER BY character_id"
    ).fetchall()
    return [row["character_id"] for row in rows]


# Compatibility alias for older imports and docs. The implementation now refreshes
# user_memory_sections, not full per-character prompt rows.
def refresh_prompt_caches_after_fact_update(
    conn: Any,
    *,
    user_id: str,
    results: list[dict[str, Any]],
    active_character_id: str | None = None,
    session_id: str | None = None,
) -> list[str]:
    return refresh_memory_sections_after_fact_update(
        conn,
        user_id=user_id,
        results=results,
        active_character_id=active_character_id,
        session_id=session_id,
    )


class CharacterNotFound(RuntimeError):
    pass


class UserNotFound(RuntimeError):
    pass
