from __future__ import annotations

import re
from typing import Any

from eval.character_dataset import CharacterEvalCase


COOKING_KEYWORDS = {
    "요리", "식사", "재료", "레시피", "조리", "냉장고", "식단", "메뉴", "저녁", "장보기"
}
STUDY_KEYWORDS = {
    "공부", "학습", "과제", "시험", "발표", "졸업작품", "논문", "평가", "지표", "프로젝트"
}


def domain_recall_accuracy(
    rows: list[dict[str, Any]],
    expected_external_ids: tuple[str, ...],
) -> float:
    if not expected_external_ids:
        return 1.0
    returned = {row.get("external_id") for row in rows}
    return 1.0 if returned.intersection(expected_external_ids) else 0.0


def memory_contamination_rate(
    rows: list[dict[str, Any]],
    forbidden_owner: str,
) -> dict[str, float | int]:
    private_rows = [
        row for row in rows
        if (row.get("scope") or "shared") == "character_private"
    ]
    leaked = [
        row for row in private_rows
        if row.get("owner_character_id") == forbidden_owner
    ]
    denominator = len(private_rows) or 1
    return {
        "leaked_private": len(leaked),
        "private_total": len(private_rows),
        "mcr": len(leaked) / denominator,
    }


def estimate_prompt_tokens(rows: list[dict[str, Any]]) -> int:
    text = "\n".join(
        str(row.get("summary_for_prompt") or row.get("content") or "")
        for row in rows
    )
    return max(0, round(len(text) / 4))


def predict_switch(case: CharacterEvalCase) -> dict[str, Any]:
    query_tokens = set(re.findall(r"[0-9A-Za-z가-힣]+", case.query.lower()))
    cooking_score = keyword_score(query_tokens, COOKING_KEYWORDS)
    study_score = keyword_score(query_tokens, STUDY_KEYWORDS)

    target = None
    score = 0
    if cooking_score > study_score and case.active_character_id != "cooking":
        target = "cooking"
        score = cooking_score
    elif study_score > cooking_score and case.active_character_id != "study":
        target = "study"
        score = study_score

    switched_to = target if target and score >= 7 else None
    expected = case.expected_switch_to
    return {
        "expected_switch_to": expected,
        "predicted_switch_to": switched_to,
        "score": score,
        "correct": 1 if switched_to == expected else 0,
    }


def keyword_score(tokens: set[str], keywords: set[str]) -> int:
    hits = 0
    joined = " ".join(tokens)
    for keyword in keywords:
        if keyword in tokens or keyword in joined:
            hits += 1
    if hits == 0:
        return 0
    return min(10, 6 + hits)
