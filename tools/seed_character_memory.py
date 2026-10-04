from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.database import ensure_schema, get_conn
from server.character_memory import rebuild_all_memory_sections
from server.memory import embed




def main() -> None:
    args = parse_args()
    fixture = args.fixture.resolve()
    data = json.loads(fixture.read_text(encoding="utf-8"))

    if args.user_id:
        data.setdefault("user", {})["user_id"] = args.user_id

    stats = summarize(data)
    print_summary(fixture, data, stats)
    if args.dry_run:
        return

    ensure_schema()
    with get_conn() as conn:
        upsert_user(conn, data["user"])
        upsert_characters(conn, data.get("characters", []))
        inserted = replace_seed_memories(
            conn,
            data["user"]["user_id"],
            data.get("memories", []),
            embed_memories=args.embed,
        )
        if args.rebuild_sections:
            rebuild_all_memory_sections(conn, user_id=data["user"]["user_id"])

    print(f"Inserted memories: {inserted}")
    print(f"User ID: {data['user']['user_id']}")
    print(f"API key: {data['user']['api_key']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed study/cooking character-aware memory data into PostgreSQL."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Seed JSON path.",
    )
    parser.add_argument(
        "--user-id",
        help="Override fixture user_id before inserting.",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate OpenAI embeddings for seeded memories. Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--rebuild-sections",
        action="store_true",
        help="Rebuild user_memory_sections after inserting seed facts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize fixture without touching the database.",
    )
    return parser.parse_args()


def summarize(data: dict[str, Any]) -> dict[str, Counter[str]]:
    memories = data.get("memories", [])
    external_ids = [m.get("external_id") for m in memories]
    duplicates = sorted(
        external_id
        for external_id, count in Counter(external_ids).items()
        if external_id and count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate external_id values: {', '.join(duplicates)}")

    return {
        "scope": Counter(m.get("scope", "shared") for m in memories),
        "owner": Counter(m.get("owner_character_id") or "shared" for m in memories),
        "type": Counter(m.get("memory_type") or "unknown" for m in memories),
    }


def print_summary(
    fixture: Path,
    data: dict[str, Any],
    stats: dict[str, Counter[str]],
) -> None:
    user = data.get("user", {})
    print(f"Fixture: {fixture}")
    print(f"User: {user.get('user_id')} ({user.get('user_name')})")
    print(f"Characters: {len(data.get('characters', []))}")
    print(f"Memories: {len(data.get('memories', []))}")
    for name, counter in stats.items():
        joined = ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))
        print(f"{name}: {joined}")


def upsert_user(conn: Any, user: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO users (
            user_id, user_name, age, birth_date, persona, robot_name,
            job, living_info, location, habit, api_key
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            user_name = EXCLUDED.user_name,
            age = EXCLUDED.age,
            birth_date = EXCLUDED.birth_date,
            persona = EXCLUDED.persona,
            robot_name = EXCLUDED.robot_name,
            job = EXCLUDED.job,
            living_info = EXCLUDED.living_info,
            location = EXCLUDED.location,
            habit = EXCLUDED.habit,
            api_key = EXCLUDED.api_key
        """,
        (
            user["user_id"],
            user["user_name"],
            user.get("age"),
            parse_date(user.get("birth_date")),
            user.get("persona"),
            user.get("robot_name"),
            user.get("job"),
            user.get("living_info"),
            user.get("location"),
            user.get("habit"),
            user["api_key"],
        ),
    )


def upsert_characters(conn: Any, characters: list[dict[str, Any]]) -> None:
    for character in characters:
        conn.execute(
            """
            INSERT INTO characters (
                character_id, name, description, system_prompt, switch_rules,
                allowed_domains, blocked_domains, tone
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (character_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                system_prompt = EXCLUDED.system_prompt,
                switch_rules = EXCLUDED.switch_rules,
                allowed_domains = EXCLUDED.allowed_domains,
                blocked_domains = EXCLUDED.blocked_domains,
                tone = EXCLUDED.tone
            """,
            (
                character["character_id"],
                character["name"],
                character.get("description"),
                character.get("system_prompt"),
                character.get("switch_rules"),
                character.get("allowed_domains") or [],
                character.get("blocked_domains") or [],
                character.get("tone"),
            ),
        )


def replace_seed_memories(
    conn: Any,
    user_id: str,
    memories: list[dict[str, Any]],
    *,
    embed_memories: bool,
) -> int:
    external_ids = [m["external_id"] for m in memories if m.get("external_id")]
    if external_ids:
        conn.execute(
            """
            DELETE FROM memory_facts
            WHERE user_id = %s AND external_id = ANY(%s)
            """,
            (user_id, external_ids),
        )

    inserted = 0
    for memory in memories:
        content = memory["content"].strip()
        vector = np.asarray(embed(content), dtype=np.float32) if embed_memories else None
        conn.execute(
            """
            INSERT INTO memory_facts (
                user_id, external_id, content, summary_for_prompt, memory_type,
                scope, owner_character_id, domain_tags, importance, event_time,
                valid_until, validity_status, confidence, sensitivity,
                conflict_group_id, source_session_id, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                memory.get("external_id"),
                content,
                memory.get("summary_for_prompt") or content,
                memory.get("memory_type"),
                memory.get("scope", "shared"),
                memory.get("owner_character_id"),
                memory.get("domain_tags") or [],
                memory.get("importance"),
                parse_date(memory.get("event_time")),
                parse_date(memory.get("valid_until")),
                memory.get("validity_status", "current"),
                memory.get("confidence", 1.0),
                memory.get("sensitivity", "normal"),
                memory.get("conflict_group_id"),
                memory.get("source_session_id"),
                vector,
            ),
        )
        inserted += 1
    return inserted


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


if __name__ == "__main__":
    main()
