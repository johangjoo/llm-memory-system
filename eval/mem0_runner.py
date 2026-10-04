"""Mem0(원조 라이브러리)로 같은 데이터·질문을 돌려 우리 시스템과 비교한다.

흐름(우리가 test2에 한 것과 동일한 데이터):
1. test2의 raw_sessions(39세션)를 Mem0에 add → Mem0가 내부적으로 fact 추출·저장
2. 같은 퀴즈 10문항을 Mem0 방식(질문별 search → 관련 메모리를 컨텍스트로)으로 답변
3. 같은 judge_score(0~1)로 채점, 토큰 측정

Mem0는 gpt-5.5의 max_tokens 비호환 때문에 gpt-4o-mini로 추출·검색한다(--mem0-model).
공정성을 위해 답변·채점 모델은 우리 실험과 같게 둘 수 있다(--answer-model).

사용:
  python -m eval.mem0_runner --quiz data/lme_pick/852ce960_quiz.json --user test2
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

from eval.judge import judge_score
from eval.prompts import EVAL_SYSTEM, ask
from server.database import ensure_schema, get_conn


def load_sessions(user_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT session_date, transcript FROM raw_sessions
            WHERE user_id = %s ORDER BY session_date, session_id
            """,
            (user_id,),
        ).fetchall()
    return rows


def build_mem0(model: str):
    from mem0 import Memory
    config = {
        "llm": {"provider": "openai", "config": {"model": model}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    }
    return Memory.from_config(config)


def run(quiz_path: str, user_id: str, mem0_model: str = "gpt-4o-mini",
        top_k: int = 20) -> dict[str, Any]:
    ensure_schema()
    quiz = json.loads(Path(quiz_path).read_text(encoding="utf-8"))
    items = quiz["quiz"]
    sessions = load_sessions(user_id)
    mem_uid = f"mem0_{user_id}"

    print(f"=== Mem0 비교: {quiz_path} (user={user_id}, {len(items)}문항) ===")
    print(f"  Mem0 추출/검색 모델: {mem0_model} / 임베딩: text-embedding-3-small\n")

    m = build_mem0(mem0_model)
    # 깨끗하게: 기존 메모리 제거 후 적재
    try:
        m.delete_all(user_id=mem_uid)
    except Exception:
        pass

    # 1) 39세션 적재 (Mem0가 내부적으로 fact 추출)
    print(f"[1] {len(sessions)}세션 Mem0 적재 중...")
    for i, s in enumerate(sessions, 1):
        turns = s["transcript"] or []
        msgs = [{"role": t["role"], "content": t.get("text") or t.get("content", "")} for t in turns]
        if not msgs:
            continue
        try:
            m.add(msgs, user_id=mem_uid, metadata={"date": str(s["session_date"])})
        except Exception as e:
            print(f"   세션{i} add 실패: {str(e)[:80]}")
        if i % 10 == 0:
            print(f"   {i}/{len(sessions)} 적재")

    # 적재된 총 메모리 수
    try:
        allmem = m.get_all(filters={"user_id": mem_uid}, top_k=500).get("results", [])
        print(f"   → Mem0가 추출·저장한 메모리: {len(allmem)}개\n")
    except Exception:
        allmem = []

    # 2) 질문별 search → 컨텍스트 → 답변 → 채점
    print("[2] 10문항 채점...")
    summary = {"score_sum": 0.0, "hits": 0, "total": 0, "tokens": 0, "cached": 0, "answer_tok": 0}
    rows = []
    for it in items:
        q, gold, qtype = it["question"], it["answer"], it["type"]
        try:
            res = m.search(q, filters={"user_id": mem_uid}, top_k=top_k).get("results", [])
        except Exception as e:
            res = []
            print(f"   {it['qid']} search 실패: {str(e)[:80]}")
        mems = [r.get("memory", "") for r in res]
        block = "\n".join(f"- {x}" for x in mems) or "(관련 기억 없음)"
        msgs = [
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "system", "content": f"<관련 기억(Mem0 검색)>\n{block}"},
            {"role": "user", "content": q},
        ]
        out = ask(msgs)
        score = judge_score(q, gold, out["answer"], qtype)
        summary["score_sum"] += score
        summary["hits"] += 1 if score >= 0.5 else 0
        summary["total"] += 1
        summary["tokens"] += out["prompt_tokens"]
        summary["cached"] += out.get("cached_tokens", 0)
        summary["answer_tok"] += out["completion_tokens"]
        rows.append({
            "qid": it["qid"], "type": qtype, "question": q, "answer": gold,
            "mem0_score": score, "mem0_ans": out["answer"],
            "mem0_ptok": out["prompt_tokens"], "mem0_retrieved": len(mems),
            "mem0_mems": mems,
        })
        print(f"   [{it['qid']}] 점수 {score:.2f} | 검색 {len(mems)}개 | 입력 {out['prompt_tokens']:,} | {out['answer'][:45]}")

    n = summary["total"] or 1
    print(f"\n=== Mem0 결과 ===")
    print(f"  총점 {summary['score_sum']:.1f}/{summary['total']} ({summary['score_sum']/n:.0%})  "
          f"정답수 {summary['hits']}/{summary['total']}")
    print(f"  질문당 평균 입력토큰 {summary['tokens']/n:,.0f}  (총 {summary['tokens']:,})")
    print(f"  추출·저장 메모리 {len(allmem)}개")

    result = {
        "quiz": quiz_path, "user": user_id, "mem0_model": mem0_model,
        "mem0_total_memories": len(allmem), "summary": summary, "rows": rows,
    }
    # 저장
    from datetime import datetime
    rd = Path(__file__).resolve().parent / "results"
    rd.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = rd / f"{quiz.get('source_item','quiz')}_mem0_{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n📄 저장: {base.with_suffix('.json')}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiz", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--mem0-model", default="gpt-4o-mini")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    run(args.quiz, args.user, args.mem0_model, args.top_k)
