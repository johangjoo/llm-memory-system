"""LongMemEval(주력) 평가 러너.

기본 사용:
    from eval.runner import run_longmemeval
    run_longmemeval(limit=50, tests=("T1","T2","T3"), variant="s")

전체 500문항을 돌리면 LLM 호출이 많으니 limit으로 작게 시작 권장.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Iterable
from uuid import uuid4

from openai import APIStatusError, OpenAIError, RateLimitError
from psycopg.types.json import Jsonb

from eval.judge import judge
from eval.loaders import (
    load_longmemeval_m,
    load_longmemeval_oracle,
    load_longmemeval_s,
)
from eval.prompts import ask, build_T1, build_T2, build_T3, build_T4_instructions
from eval.realtime_ask import RealtimeAskError, ask_realtime
from server.database import ensure_schema, get_conn
from server.memory import extract_facts, store_facts


VARIANT_LOADERS = {
    "s": load_longmemeval_s,
    "m": load_longmemeval_m,
    "oracle": load_longmemeval_oracle,
}


def reset_user(conn, user_id: str) -> None:
    """동일 user_id로 다시 돌릴 때 깨끗하게(eval 전용 픽스처)."""
    conn.execute("DELETE FROM memory_facts WHERE user_id = %s", (user_id,))
    conn.execute("DELETE FROM reflections WHERE user_id = %s", (user_id,))
    conn.execute("DELETE FROM raw_sessions WHERE user_id = %s", (user_id,))


def _store_sessions(conn, user_id: str, sessions: list[dict[str, Any]]) -> list[tuple[str, Any, str]]:
    """sessions를 raw_sessions에 적재하고 (sid, date, raw_text) 목록 반환."""
    written: list[tuple[str, Any, str]] = []
    for index, session in enumerate(sessions):
        sid = f"{user_id}_sess{index}"
        turns = session.get("turns", [])
        raw_text = "\n".join(f"{t['role']}: {t['text']}" for t in turns)
        session_date = _parse_date(session.get("date"))
        conn.execute(
            """
            INSERT INTO raw_sessions (session_id, user_id, session_date, transcript, raw_text)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            (sid, user_id, session_date, Jsonb(turns), raw_text),
        )
        written.append((sid, session_date, raw_text))
    return written


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip().replace("/", "-")
        if len(text) >= 10:
            try:
                return date.fromisoformat(text[:10])
            except ValueError:
                return None
    return None


def _log_result(
    conn,
    *,
    test_name: str,
    dataset: str,
    item: dict[str, Any],
    user_id: str,
    answer: str,
    correct: int,
    prompt_tokens: int,
    completion_tokens: int,
    latency: float,
) -> None:
    conn.execute(
        """
        INSERT INTO eval_results (
            test_name, dataset, question_id, question_type, user_id,
            model_answer, gold_answer, correct,
            prompt_tokens, completion_tokens, latency_sec
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            test_name, dataset, item["question_id"], item["question_type"], user_id,
            answer, item["answer"], correct,
            prompt_tokens, completion_tokens, latency,
        ),
    )


def run_longmemeval(
    limit: int = 10,
    tests: Iterable[str] = ("T1", "T2", "T3"),
    variant: str = "s",
    k: int = 5,
    pool: int = 20,
    extract_each_session: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """LongMemEval(variant)에서 limit개 문항을 T1/T2/T3로 평가하고 eval_results에 적립.

    반환: {processed, by_test: {T1: {correct, total}, ...}}
    """
    ensure_schema()
    loader = VARIANT_LOADERS[variant]
    items = loader(limit=limit)
    dataset = f"longmemeval_{variant}"

    by_test: dict[str, dict[str, int]] = {t: {"correct": 0, "total": 0} for t in tests}
    processed = 0
    skipped: list[dict[str, Any]] = []  # 항목별 실패 기록 — quota/네트워크 등

    for index, item in enumerate(items, 1):
        user_id = item["question_id"]
        if verbose:
            print(f"[{index}/{len(items)}] {user_id} ({item['question_type']})")
        try:
            # 각 item을 별도 트랜잭션으로 — 도중 실패해도 앞 진행분 보존
            with get_conn() as conn:
                reset_user(conn, user_id)

                # WRITE: raw_sessions 적재 + (옵션) fact 추출
                stored = _store_sessions(conn, user_id, item["sessions"])
                if extract_each_session and "T3" in tests:
                    for sid, sdate, raw_text in stored:
                        facts = extract_facts(raw_text, sdate)
                        if facts:
                            store_facts(conn, user_id, sid, sdate, facts)

                # 각 테스트 실행
                for test_name in tests:
                    if test_name == "T1":
                        out = ask(build_T1(item))
                    elif test_name == "T2":
                        out = ask(build_T2(item))
                    elif test_name == "T3":
                        out = ask(build_T3(item, conn, user_id, k=k, pool=pool))
                    elif test_name == "T4":
                        instructions, question = build_T4_instructions(
                            item, conn, user_id, k=k, pool=pool
                        )
                        out = ask_realtime(instructions, question)
                    else:
                        continue
                    correct = judge(item["question"], item["answer"], out["answer"], item["question_type"])
                    _log_result(
                        conn,
                        test_name=test_name, dataset=dataset, item=item, user_id=user_id,
                        answer=out["answer"], correct=correct,
                        prompt_tokens=out["prompt_tokens"], completion_tokens=out["completion_tokens"],
                        latency=out["latency"],
                    )
                    by_test[test_name]["correct"] += correct
                    by_test[test_name]["total"] += 1
                    if verbose:
                        print(f"   {test_name}  correct={correct}  tokens={out['prompt_tokens']}+{out['completion_tokens']}")
            # with get_conn() 종료 시점에 트랜잭션 commit — 이 item의 결과가 DB에 보존됨
            processed += 1
        except RateLimitError as exc:
            # quota/속도 한도 — 더 시도해도 같은 결과. 여기서 중단하고 진행분 보고.
            print(f"   [중단] RateLimitError: {exc}")
            skipped.append({"question_id": user_id, "reason": "rate_limit_or_quota", "detail": str(exc)})
            break
        except (APIStatusError, OpenAIError) as exc:
            # 일시적 API 오류 — 해당 item만 건너뛰고 다음으로
            print(f"   [skip] OpenAI 오류: {exc}")
            skipped.append({"question_id": user_id, "reason": "openai_error", "detail": str(exc)})
            continue
        except Exception as exc:
            print(f"   [skip] 예상치 못한 오류: {exc!r}")
            skipped.append({"question_id": user_id, "reason": "other", "detail": repr(exc)})
            continue

    return {
        "processed": processed,
        "attempted": len(items),
        "skipped": skipped,
        "by_test": by_test,
        "dataset": dataset,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--variant", default="s", choices=list(VARIANT_LOADERS))
    ap.add_argument("--tests", default="T1,T2,T3")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    tests = tuple(t.strip() for t in args.tests.split(",") if t.strip())
    summary = run_longmemeval(
        limit=args.limit,
        tests=tests,
        variant=args.variant,
        k=args.k,
        pool=args.pool,
        extract_each_session=not args.no_extract,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
