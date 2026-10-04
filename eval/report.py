"""eval_results 집계 리포트."""

from __future__ import annotations

import argparse
import json

from server.database import ensure_schema, get_conn


def report(dataset: str | None = None) -> dict[str, object]:
    ensure_schema()  # 풀이 열려있지 않을 수 있으니 보장
    where = ""
    params: tuple = ()
    if dataset:
        where = "WHERE dataset = %s"
        params = (dataset,)
    with get_conn() as conn:
        # 정확도 (test x question_type)
        rows = conn.execute(
            f"""
            SELECT test_name, question_type,
                   AVG(correct)::float AS acc, COUNT(*) AS n,
                   AVG(prompt_tokens)::float AS avg_prompt_tok,
                   AVG(completion_tokens)::float AS avg_comp_tok,
                   AVG(latency_sec)::float AS avg_latency
            FROM eval_results {where}
            GROUP BY test_name, question_type
            ORDER BY question_type, test_name
            """,
            params,
        ).fetchall()
        # 환각(abstention인데 T1이 틀린 케이스)
        halluc = conn.execute(
            f"""
            SELECT question_id, model_answer
            FROM eval_results
            WHERE test_name='T1' AND question_type='abstention' AND correct=0
              {("AND dataset = %s" if dataset else "")}
            """,
            params,
        ).fetchall()
    return {
        "by_axis": [dict(r) for r in rows],
        "t1_abstention_hallucinations": [dict(r) for r in halluc],
    }


def print_report(dataset: str | None = None) -> None:
    data = report(dataset)
    print(f"=== 결과 (dataset={dataset or '전체'}) ===")
    print(f"{'qtype':24} {'test':5} {'acc':>6} {'n':>4} {'p_tok':>8} {'c_tok':>6} {'lat':>6}")
    for r in data["by_axis"]:
        print(
            f"{(r['question_type'] or '-'):24} {r['test_name']:5} "
            f"{r['acc']:6.3f} {r['n']:4d} {r['avg_prompt_tok']:8.0f} "
            f"{r['avg_comp_tok']:6.0f} {r['avg_latency']:6.2f}"
        )
    if data["t1_abstention_hallucinations"]:
        print(f"\n환각 사례(T1 abstention 오답): {len(data['t1_abstention_hallucinations'])}건")
        for h in data["t1_abstention_hallucinations"][:5]:
            print(f"  - {h['question_id']}: {(h['model_answer'] or '')[:80]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="예: longmemeval_s")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.json:
        print(json.dumps(report(args.dataset), ensure_ascii=False, indent=2, default=str))
    else:
        print_report(args.dataset)
