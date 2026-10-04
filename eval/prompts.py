"""T1/T2/T3 메시지 빌더 + 응답 생성 호출."""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from server.memory import get_client, retrieve_ranked


MODEL_GENERATION = os.getenv("MINDMATE_GENERATION_MODEL", "gpt-5.5")

EVAL_SYSTEM = (
    "너는 사용자의 과거 대화에서 필요한 정보만 근거로 답하는 평가용 AI다.\n"
    "- <관련 기억> 또는 대화 이력에 없는 내용은 지어내지 않는다.\n"
    "- 정보가 바뀌었으면 최신 정보를 우선한다.\n"
    "- 한국어/영어는 사용자가 묻는 언어에 맞춰 간결하게 답한다."
)


def build_T1(item: dict[str, Any]) -> list[dict[str, str]]:
    """기억 OFF — system + 질문만."""
    return [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "user", "content": item["question"]},
    ]


def build_T2(item: dict[str, Any]) -> list[dict[str, str]]:
    """full-context — 세션 turn 전부 펼침."""
    messages: list[dict[str, str]] = [{"role": "system", "content": EVAL_SYSTEM}]
    for session in item.get("sessions", []):
        date = session.get("date")
        if date:
            messages.append({"role": "system", "content": f"[{date} 대화]"})
        for turn in session.get("turns", []):
            messages.append({"role": turn["role"], "content": turn["text"]})
    messages.append({"role": "user", "content": item["question"]})
    return messages


def build_T3(item: dict[str, Any], conn, user_id: str, k: int = 5, pool: int = 20) -> list[dict[str, str]]:
    """우리 시스템 — 가중 검색으로 뽑은 fact 블록만 주입."""
    rows = retrieve_ranked(
        conn,
        user_id,
        item["question"],
        k=k,
        pool=pool,
        include_prompt_excluded=True,
    )
    if rows:
        block = "\n".join(
            f"- {row['content']} (출처: {row.get('event_time') or '-'})" for row in rows
        )
    else:
        block = "(관련 기억 없음)"
    return [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "system", "content": f"<관련 기억>\n{block}"},
        {"role": "user", "content": item["question"]},
    ]


def build_T4_instructions(
    item: dict[str, Any], conn, user_id: str, k: int = 5, pool: int = 20
) -> tuple[str, str]:
    """T4 (Realtime) — Realtime 세션의 instructions 문자열과 질문 텍스트를 반환.

    구조: EVAL_SYSTEM + ## Context > ### Recalled Memory (T3 식 fact 블록).
    클라이언트가 session.update의 instructions에 그대로 사용하고, 별도로
    conversation.item.create에 question을 보낸다.
    """
    rows = retrieve_ranked(
        conn,
        user_id,
        item["question"],
        k=k,
        pool=pool,
        include_prompt_excluded=True,
    )
    if rows:
        block = "\n".join(
            f"- {row['content']} (출처: {row.get('event_time') or '-'})" for row in rows
        )
    else:
        block = "(관련 기억 없음)"
    instructions = EVAL_SYSTEM + "\n\n## Context\n### Recalled Memory\n" + block
    return instructions, item["question"]


def ask(messages: list[dict[str, str]], model: str | None = None) -> dict[str, Any]:
    """LLM 호출 + 사용량/지연 측정.

    반환: {answer, prompt_tokens, cached_tokens, completion_tokens, latency}.
    cached_tokens = 프롬프트 캐싱으로 할인된(이미 캐시된) 입력 토큰 수.
    """
    client = get_client()
    started = time.time()
    response = client.chat.completions.create(
        model=model or MODEL_GENERATION,
        messages=messages,
    )
    usage = response.usage
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "answer": response.choices[0].message.content or "",
        "prompt_tokens": usage.prompt_tokens,
        "cached_tokens": cached,
        "completion_tokens": usage.completion_tokens,
        "latency": time.time() - started,
    }
