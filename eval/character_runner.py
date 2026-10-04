from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.character_dataset import CharacterEvalCase, get_cases
from eval.character_metrics import (
    domain_recall_accuracy,
    estimate_prompt_tokens,
    memory_contamination_rate,
    predict_switch,
)
from eval.character_report import write_report
from server.database import ensure_schema, get_conn
from server.memory_policy import apply_memory_policy


BASELINES = ("no_memory", "all_memory", "tag_rag", "mindmate_map")


def run_character_eval(
    *,
    user_id: str = "user_default",
    limit: int | None = None,
    pack_shared_limit: int = 8,
    pack_private_limit: int = 8,
    tag_limit: int = 16,
    output_dir: Path = Path("eval/results"),
) -> dict[str, Any]:
    ensure_schema()
    cases = get_cases(limit)

    with get_conn() as conn:
        all_rows = fetch_current_memories(conn, user_id)
        result_cases: list[dict[str, Any]] = []
        memory_cases = [case for case in cases if case.memory_eval]
        accum: dict[str, dict[str, float]] = {
            baseline: {
                "dra": 0.0,
                "mcr": 0.0,
                "prompt_tokens": 0.0,
                "memory_count": 0.0,
                "leaked_private": 0.0,
                "private_total": 0.0,
            }
            for baseline in BASELINES
        }
        switch_correct = 0

        for case in cases:
            case_baselines: dict[str, Any] = {}
            if case.memory_eval:
                for baseline in BASELINES:
                    rows = select_rows(
                        conn,
                        baseline=baseline,
                        user_id=user_id,
                        case=case,
                        all_rows=all_rows,
                        pack_shared_limit=pack_shared_limit,
                        pack_private_limit=pack_private_limit,
                        tag_limit=tag_limit,
                    )
                    leak = memory_contamination_rate(rows, case.forbidden_owner)
                    dra = domain_recall_accuracy(rows, case.expected_external_ids)
                    prompt_tokens = estimate_prompt_tokens(rows)
                    metrics = {
                        "dra": dra,
                        "mcr": leak["mcr"],
                        "leaked_private": leak["leaked_private"],
                        "private_total": leak["private_total"],
                        "prompt_tokens": prompt_tokens,
                        "memory_count": len(rows),
                        "external_ids": [
                            row["external_id"] for row in rows if row.get("external_id")
                        ],
                    }
                    case_baselines[baseline] = metrics
                    accum[baseline]["dra"] += float(dra)
                    accum[baseline]["mcr"] += float(leak["mcr"])
                    accum[baseline]["prompt_tokens"] += float(prompt_tokens)
                    accum[baseline]["memory_count"] += float(len(rows))
                    accum[baseline]["leaked_private"] += float(leak["leaked_private"])
                    accum[baseline]["private_total"] += float(leak["private_total"])

            switch = predict_switch(case)
            switch_correct += int(switch["correct"])
            result_cases.append({
                "case_id": case.case_id,
                "active_character_id": case.active_character_id,
                "query": case.query,
                "expected_external_ids": list(case.expected_external_ids),
                "forbidden_owner": case.forbidden_owner,
                "memory_eval": case.memory_eval,
                "switch": switch,
                "baselines": case_baselines,
            })

    case_count = len(cases) or 1
    memory_case_count = len(memory_cases) or 1
    summary = {
        baseline: {
            "dra": values["dra"] / memory_case_count,
            "mcr": values["mcr"] / memory_case_count,
            "prompt_tokens": values["prompt_tokens"] / memory_case_count,
            "memory_count": values["memory_count"] / memory_case_count,
            "leaked_private": int(values["leaked_private"]),
            "private_total": int(values["private_total"]),
        }
        for baseline, values in accum.items()
    }
    result = {
        "user_id": user_id,
        "case_count": len(cases),
        "memory_case_count": len(memory_cases),
        "baselines": list(BASELINES),
        "summary": summary,
        "switch_summary": {
            "correct": switch_correct,
            "total": len(cases),
            "accuracy": switch_correct / case_count,
        },
        "cases": result_cases,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"character_eval_{timestamp}.json"
    md_path = output_dir / f"character_eval_{timestamp}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result, md_path)
    result["json_path"] = str(json_path)
    result["md_path"] = str(md_path)
    return result


def fetch_current_memories(conn: Any, user_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT memory_id, external_id, user_id, content, summary_for_prompt,
               memory_type, scope, owner_character_id, domain_tags, importance,
               event_time, valid_until, validity_status, confidence, sensitivity,
               conflict_group_id, source_session_id, created_at
        FROM memory_facts
        WHERE user_id = %s
          AND (valid_until IS NULL OR valid_until > CURRENT_DATE)
        ORDER BY COALESCE(event_time, created_at::date) DESC NULLS LAST,
                 importance DESC NULLS LAST,
                 memory_id DESC
        """,
        (user_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def select_rows(
    conn: Any,
    *,
    baseline: str,
    user_id: str,
    case: CharacterEvalCase,
    all_rows: list[dict[str, Any]],
    pack_shared_limit: int,
    pack_private_limit: int,
    tag_limit: int,
) -> list[dict[str, Any]]:
    if baseline == "no_memory":
        return []
    if baseline == "all_memory":
        return list(all_rows)
    if baseline == "tag_rag":
        return simple_tag_rag(all_rows, case.query, limit=tag_limit)
    if baseline == "mindmate_map":
        character = conn.execute(
            """
            SELECT character_id, name, description, allowed_domains, blocked_domains, tone
            FROM characters WHERE character_id = %s
            """,
            (case.active_character_id,),
        ).fetchone()
        allowed = apply_memory_policy(
            conn,
            user_id=user_id,
            active_character_id=case.active_character_id,
            memories=all_rows,
            character=character,
            query=case.query,
            log_decisions=False,
        )
        shared = [
            row for row in allowed
            if (row.get("scope") or "shared") == "shared"
        ][:pack_shared_limit]
        private = [
            row for row in allowed
            if (row.get("scope") or "shared") == "character_private"
            and row.get("owner_character_id") == case.active_character_id
        ][:pack_private_limit]
        return shared + private
    raise ValueError(f"unknown baseline: {baseline}")


def simple_tag_rag(
    rows: list[dict[str, Any]],
    query: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    query_tokens = set(re.findall(r"[0-9A-Za-z가-힣]+", query.lower()))
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        haystack_parts = [
            str(row.get("summary_for_prompt") or ""),
            str(row.get("content") or ""),
            " ".join(row.get("domain_tags") or []),
            str(row.get("owner_character_id") or ""),
            str(row.get("memory_type") or ""),
        ]
        haystack = " ".join(haystack_parts).lower()
        overlap = sum(1 for token in query_tokens if token in haystack)
        importance = float(row.get("importance") or 3) / 5.0
        score = overlap + importance
        if overlap > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default="user_default")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args()
    result = run_character_eval(
        user_id=args.user_id,
        limit=args.limit,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "summary": result["summary"],
        "switch_summary": result["switch_summary"],
        "json_path": result["json_path"],
        "md_path": result["md_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
