"""한 인물(user_id)에 대한 다중 질문 퀴즈 채점기.

LongMemEval은 항목당 질문이 1개라, 한 인물을 여러 각도로 보려면 직접 만든
퀴즈 세트(data/lme_pick/<id>_quiz.json)를 쓴다. 이 스크립트는 그 퀴즈를
T1/T2/T3 세 조건으로 돌려 LLM-as-judge로 채점하고 표를 출력한다.

- T1 (기억 OFF): system + 질문만
- T2 (full-context): 그 user의 raw_sessions 원문 전부 + 질문
- T3 (우리 시스템): 그 user의 user_memory_sections 압축본 + 질문

사용:
    python -m eval.quiz_runner --quiz data/lme_pick/852ce960_quiz.json --user test2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.judge import judge_score
from eval.prompts import EVAL_SYSTEM, ask
from server.database import ensure_schema, get_conn


def load_user_context(user_id: str) -> tuple[str, str | None]:
    """T2용 raw_sessions 원문 전체와 T3용 section cache를 가져온다."""
    with get_conn() as conn:
        sessions = conn.execute(
            """
            SELECT session_date, raw_text FROM raw_sessions
            WHERE user_id = %s ORDER BY session_date, session_id
            """,
            (user_id,),
        ).fetchall()
        section_rows = conn.execute(
            """
            SELECT section_key, section_text
            FROM user_memory_sections
            WHERE user_id = %s
            ORDER BY CASE section_key
                WHEN 'shared' THEN 0
                WHEN 'study' THEN 1
                WHEN 'cooking' THEN 2
                ELSE 3
            END
            """,
            (user_id,),
        ).fetchall()
    full_context = "\n\n".join(
        f"[{s['session_date']} 대화]\n{s['raw_text']}" for s in sessions
    )
    memory_sections_text = "\n\n".join(
        f"## {row['section_key']}\n{row['section_text']}"
        for row in section_rows
        if row.get("section_text")
    ) or None
    return full_context, memory_sections_text


def build_messages(mode: str, question: str, full_context: str, memory_sections_text: str | None):
    if mode == "T1":
        return [
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "user", "content": question},
        ]
    if mode == "T2":
        return [
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "system", "content": f"<과거 대화 전체>\n{full_context}"},
            {"role": "user", "content": question},
        ]
    if mode == "T3":
        block = memory_sections_text or "(저장된 메모리 없음)"
        return [
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "system", "content": f"<사용자 장기기억>\n{block}"},
            {"role": "user", "content": question},
        ]
    raise ValueError(mode)


def _base_history(mode: str, full_context: str, memory_sections_text: str | None):
    """모드별 '깔아두는' 시스템 메시지(원문 또는 정리본)."""
    if mode == "T1":
        return [{"role": "system", "content": EVAL_SYSTEM}]
    if mode == "T2":
        return [{"role": "system", "content": EVAL_SYSTEM},
                {"role": "system", "content": f"<과거 대화 전체>\n{full_context}"}]
    if mode == "T3":
        block = memory_sections_text or "(저장된 메모리 없음)"
        return [{"role": "system", "content": EVAL_SYSTEM},
                {"role": "system", "content": f"<사용자 장기기억>\n{block}"}]
    raise ValueError(mode)


def run_quiz(quiz_path: str, user_id: str, tests=("T1", "T2", "T3")) -> dict[str, Any]:
    """멀티턴 구조: 모드별로 원문/정리본을 한 번 깔고 질문 10개를 대화처럼 이어간다.

    측정: 첫 프롬프트 토큰(원문/정리본이 처음 들어가는 양) + 질문별
    입력토큰(캐시/새로) + 답변토큰 + 점수.
    """
    ensure_schema()
    quiz = json.loads(Path(quiz_path).read_text(encoding="utf-8"))
    items = quiz["quiz"]
    full_context, memory_sections_text = load_user_context(user_id)

    print(f"=== 퀴즈 채점(멀티턴): {quiz_path} (user={user_id}, {len(items)}문항) ===")
    print(f"  T2 원문 {len(full_context):,}자 / T3 정리본 {len(memory_sections_text or ''):,}자\n")

    summary = {t: {"score_sum": 0.0, "hits": 0, "total": 0,
                   "first_prompt_tok": 0, "qa_new_tok": 0, "qa_cached_tok": 0,
                   "answer_tok": 0, "tokens": 0, "cached": 0} for t in tests}
    rows_by_q = {it["qid"]: {"qid": it["qid"], "type": it["type"],
                             "question": it["question"], "answer": it["answer"],
                             "grading_note": it.get("grading_note", "")} for it in items}

    for mode in tests:
        print(f"───── {mode} 멀티턴 시작 ─────")
        history = _base_history(mode, full_context, memory_sections_text)
        for i, it in enumerate(items):
            q, gold, qtype = it["question"], it["answer"], it["type"]
            history.append({"role": "user", "content": q})
            out = ask(history)
            score = judge_score(q, gold, out["answer"], qtype)
            history.append({"role": "assistant", "content": out["answer"]})

            ptok = out["prompt_tokens"]; cached = out.get("cached_tokens", 0)
            new_in = ptok - cached; atok = out["completion_tokens"]
            s = summary[mode]
            if i == 0:
                s["first_prompt_tok"] = ptok          # 첫 프롬프트(원문/정리본 처음)
            else:
                s["qa_new_tok"] += new_in             # 2번째 질문부터 새로 드는 입력
                s["qa_cached_tok"] += cached
            s["answer_tok"] += atok
            s["score_sum"] += score
            s["hits"] += 1 if score >= 0.5 else 0
            s["total"] += 1
            s["tokens"] += ptok
            s["cached"] += cached

            r = rows_by_q[it["qid"]]
            r[mode] = score
            r[f"{mode}_ans"] = out["answer"]
            r[f"{mode}_ptok"] = ptok
            r[f"{mode}_cached"] = cached
            r[f"{mode}_new"] = new_in
            r[f"{mode}_atok"] = atok
            print(f"  [{it['qid']}] 점수 {score:.2f} | 입력 {ptok:>7,}(캐시 {cached:>7,}/새 {new_in:>6,}) | 답변 {atok:>4} tok")
        print()

    rows = list(rows_by_q.values())
    print(_dashboard(summary, tests))

    result = {
        "quiz": quiz_path, "user": user_id, "tests": list(tests),
        "summary": summary, "rows": rows,
        "t2_context_chars": len(full_context),
        "t3_prompt": memory_sections_text or "",
        "t3_prompt_chars": len(memory_sections_text or ""),
    }
    _save_result(quiz, user_id, tests, result)
    return result


RESULTS_DIR = Path(__file__).resolve().parent / "results"

# gpt-5.5 가정 단가(USD per 1M tokens). 실제 단가에 맞게 조정.
PRICE_INPUT = 1.25       # 일반 입력
PRICE_CACHED = 0.125     # 캐시된 입력(약 1/10 가정)


def _bar(value: float, vmax: float, width: int = 40) -> str:
    if vmax <= 0:
        return ""
    return "█" * max(1, round(width * value / vmax)) if value > 0 else ""


def _cost(s: dict) -> float:
    """추정 입력비용(USD). 캐시토큰은 할인 단가 적용 + 답변 토큰은 출력 단가(가정 입력과 동일)."""
    uncached = s["tokens"] - s.get("cached", 0)
    return (uncached * PRICE_INPUT + s.get("cached", 0) * PRICE_CACHED
            + s.get("answer_tok", 0) * PRICE_INPUT) / 1_000_000


def _dashboard(summary: dict, tests) -> str:
    out = ["┌─ 대시보드 (멀티턴: 원문/정리본 한 번 깔고 질문 10개) ──────────────"]

    # 1) 정확도
    out.append("│ ▸ 정확도")
    for t in tests:
        acc = summary[t]["score_sum"] / (summary[t]["total"] or 1)
        out.append(f"│   {t} {_bar(acc,1.0):36} {acc*100:5.1f}%  "
                   f"({summary[t]['score_sum']:.1f}/{summary[t]['total']})")

    # 2) 첫 프롬프트 토큰 (원문/정리본이 처음 들어가는 양)
    out.append("│\n│ ▸ ① 첫 프롬프트 토큰 (원문/정리본, 1회)")
    fmax = max(summary[t]["first_prompt_tok"] for t in tests) or 1
    for t in tests:
        v = summary[t]["first_prompt_tok"]
        out.append(f"│   {t} {_bar(v,fmax):36} {v:>9,}")

    # 3) 질답에 실제로 새로 드는 토큰 (2번째 질문부터 입력 새토큰 + 답변)
    out.append("│\n│ ▸ ② 질답 토큰 — 질문2~끝 '새로 든 입력' + 답변(캐시 제외)")
    qmax = max(summary[t]["qa_new_tok"] + summary[t]["answer_tok"] for t in tests) or 1
    for t in tests:
        new = summary[t]["qa_new_tok"]; ans = summary[t]["answer_tok"]
        out.append(f"│   {t} {_bar(new+ans,qmax):36} 새입력 {new:>6,} + 답변 {ans:>5,}")

    # 4) 캐시 & 실비용
    out.append("│\n│ ▸ ③ 캐시 적용 & 추정비용(USD)")
    out.append(f"│   {'조건':3} {'첫프롬':>8} {'질답새토큰':>9} {'캐시토큰':>10} {'추정비용':>9}")
    costs = {t: _cost(summary[t]) for t in tests}
    for t in tests:
        s = summary[t]
        out.append(f"│   {t:3} {s['first_prompt_tok']:>8,} {s['qa_new_tok']:>9,} "
                   f"{s['cached']:>10,} {costs[t]:>8.4f}$")

    if "T2" in summary and "T3" in summary and costs.get("T3"):
        cr = costs["T2"] / costs["T3"]
        out.append("│\n│ ▶ 추정비용: T2 / T3 = {:.1f}배".format(cr))
    out.append("└────────────────────────────────────────────────────────────")
    out.append("  (비용은 가정 단가 추정치 — 입력 $1.25/1M, 캐시 $0.125/1M)")
    return "\n".join(out)


TEST_LABELS = {
    "T1": "기억 없음 (질문만)",
    "T2": "원문 전부 주입 (baseline·상한선)",
    "T3": "우리 시스템 — 압축 메모리 프롬프트",
    "T4": "Realtime 음성",
}


def _save_result(quiz: dict, user_id: str, tests, result: dict) -> None:
    """채점 결과를 eval/results/ 에 json + 발표용 md로 저장한다."""
    from datetime import datetime
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = RESULTS_DIR / f"{quiz.get('source_item','quiz')}_{user_id}_{'-'.join(tests)}_{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    s = result["summary"]
    L: list[str] = []
    L += [f"# MindMate 장기기억 실험 결과 — 발표 자료", ""]
    L += [f"- **데이터**: LongMemEval-S 항목 `{quiz.get('source_item','?')}` "
          f"({quiz.get('source_dataset','')}), 한 인물의 다중 세션 대화",
          f"- **대상 user**: `{user_id}`  |  **실행 시각**: {stamp}",
          f"- **모델**: 생성·채점 모두 gpt-5.5  |  **채점**: LLM-as-judge 부분점수(0.0~1.0)",
          ""]

    # 1. 비교 조건
    L += ["## 1. 비교 조건", "",
          "| 조건 | 무엇을 LLM에 넣나 |",
          "|---|---|"]
    for t in tests:
        L.append(f"| **{t}** | {TEST_LABELS.get(t, t)} |")
    L.append("")

    # 2. 입력 규모 대비
    L += ["## 2. 입력 규모 — 원문 vs 우리 압축", "",
          f"- **원문 전체(T2가 처음 한 번 까는 것)**: {result['t2_context_chars']:,}자",
          f"- **우리 압축 메모리 프롬프트(T3)**: {result['t3_prompt_chars']:,}자",
          "",
          "측정 구조: **멀티턴** — 원문/정리본을 처음 한 번 깔고, 그 위에 질문 10개를 대화처럼 이어감.",
          "→ ① 첫 프롬프트(원문/정리본) + ② 질문2~끝 새로 드는 입력 + 답변, 을 분리 측정.", ""]

    # 3. 우리 압축 프롬프트 전문
    L += ["## 3. 우리 시스템이 만든 압축 메모리 프롬프트 (T3 입력 전문)", "",
          "```", result["t3_prompt"].rstrip(), "```", ""]

    # 4. 채점 방법
    L += ["## 4. 질문 선정·채점 방법", "",
          "- 질문은 LongMemEval의 5축(single-session / multi-session / temporal / "
          "knowledge-update / abstention)을 본떠, 해당 인물의 원문에서 **정답이 명확한 사실**만 골라 작성.",
          "- 각 질문에 정답을 미리 확정(원문 근거 `evidence_session` 기록).",
          "- 모델 답변을 judge LLM이 정답과 비교해 **0.0~1.0 부분점수**로 채점 "
          "(완전일치 1.0 / 핵심맞고 불완전 0.5~0.9 / 무관 0.0).",
          "- abstention 문항은 '모른다'류로 답하면 1.0, 지어내면 0.0.",
          ""]

    # 5. 핵심 표: 첫 프롬프트 · 질답 토큰 · 점수
    L += ["## 5. 첫 프롬프트 · 질답 토큰(캐시 구분) · 점수", "",
          "| 조건 | ① 첫 프롬프트 토큰 | ② 질답 새 입력(질문2~끝) | 질답 캐시 입력 | 답변 토큰 합 | ③ 점수 | 추정비용(USD) |",
          "|---|---|---|---|---|---|---|"]
    for t in tests:
        st = s[t]; n = st["total"] or 1
        L.append(
            f"| {t} | {st['first_prompt_tok']:,} | {st['qa_new_tok']:,} | {st['qa_cached_tok']:,} "
            f"| {st['answer_tok']:,} | **{st['score_sum']:.1f}/{st['total']} ({st['score_sum']/n:.0%})** "
            f"| {_cost(st):.4f} |"
        )
    L.append("")
    L += ["```", _dashboard(s, tests), "```", "",
          "**읽는 법**: T2는 첫 질문에 원문을 통째로 깔고(① 큰 값), 이후엔 그 원문이 캐시되어 "
          "질문당 새로 드는 입력(②)은 작다. T3는 정리본이 작아 ①이 처음부터 작다. "
          "질문이 **연속**일 땐 캐시 덕에 T2 ②가 작지만, **띄엄띄엄/장기 누적**이면 캐시가 만료·팽창해 "
          "T2가 매번 ①을 다시 부담한다.", ""]

    # 6. 문항별 전문 (질문/정답/각 조건 답변 전문+점수+토큰)
    L += ["## 6. 문항별 상세 (질문 · 정답 · 답변 전문 · 점수)", ""]
    for r in result["rows"]:
        L += [f"### [{r['qid']}] ({r['type']})", "",
              f"- **질문**: {r['question'] if 'question' in r else ''}",
              f"- **정답**: {r['answer']}"]
        if r.get("grading_note"):
            L.append(f"- 채점 기준: {r['grading_note']}")
        L.append("")
        for t in tests:
            L += [f"**{t} — 점수 {r.get(t,0):.2f}  (입력 {r.get(t+'_ptok',0):,} / 새 {r.get(t+'_new',0):,} / 답변 {r.get(t+'_atok',0):,} 토큰)**", "",
                  f"> {r.get(t+'_ans','').strip()}", ""]
    base.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n📄 발표용 결과 저장:\n  {base.with_suffix('.md')}\n  {base.with_suffix('.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiz", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--tests", default="T1,T2,T3")
    args = ap.parse_args()
    tests = tuple(t.strip() for t in args.tests.split(",") if t.strip())
    run_quiz(args.quiz, args.user, tests)
