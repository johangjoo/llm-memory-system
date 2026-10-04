"""LLM-as-judge 채점.

abstention 문항(대화에 없는 내용 묻기)은 "모른다"류 답을 정답으로 인정.
"""

from __future__ import annotations

import os

from server.memory import get_client


MODEL_JUDGE = os.getenv("MINDMATE_JUDGE_MODEL", "gpt-5.5")

JUDGE_NORMAL = """질문: {q}
정답: {gold}
모델 답: {ans}

모델 답이 정답과 의미상 일치하면 1, 아니면 0만 출력. 다른 글자 금지."""

JUDGE_ABSTAIN = """질문: {q}
이 질문은 대화에 언급되지 않은 내용을 묻는 abstention 문항이다.
모델 답: {ans}

모델 답이 "모른다 / 얘기한 적 없다 / 들은 적 없다"류로 답하면 1, 뭔가를 지어내면 0. 숫자만 출력."""


def judge(question: str, gold: str, answer: str, qtype: str) -> int:
    template = JUDGE_ABSTAIN if (qtype or "").lower() == "abstention" else JUDGE_NORMAL
    prompt = template.format(q=question, gold=gold, ans=answer)
    response = get_client().chat.completions.create(
        model=MODEL_JUDGE,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (response.choices[0].message.content or "").strip()
    return 1 if text.startswith("1") else 0


# ── 부분점수(0.0~1.0) 채점 ──────────────────────────────────────────────
JUDGE_SCORE_NORMAL = """질문: {q}
정답: {gold}
모델 답: {ans}

모델 답이 정답을 얼마나 정확히 맞혔는지 0.0~1.0 사이 점수로만 평가해라.
- 1.0: 정답과 의미가 완전히 일치(표현·단위 차이는 무관, 예: "$400,000"="사십만 달러").
- 0.5~0.9: 핵심은 맞지만 일부 부정확하거나 불완전(예: 금액 자릿수 일부만 맞음, 근접한 값).
- 0.1~0.4: 관련은 있으나 핵심이 틀림.
- 0.0: 완전히 틀리거나 무관.
숫자 하나만 출력(예: 0.7). 다른 글자 금지."""

JUDGE_SCORE_ABSTAIN = """질문: {q}
이 질문은 대화에 언급된 적 없는 내용을 묻는 abstention 문항이다.
모델 답: {ans}

모델 답이 "모른다 / 말한 적 없다 / 정보가 없다"류로 적절히 답하면 1.0,
모르겠다면서도 일부 지어내면 0.5, 확신을 갖고 사실을 지어내면 0.0으로 평가해라.
0.0~1.0 숫자 하나만 출력. 다른 글자 금지."""


def _parse_score(text: str) -> float:
    import re
    m = re.search(r"[01](?:\.\d+)?|0?\.\d+", text.strip())
    if not m:
        return 0.0
    try:
        return max(0.0, min(1.0, float(m.group())))
    except ValueError:
        return 0.0


def judge_score(question: str, gold: str, answer: str, qtype: str) -> float:
    """0.0~1.0 연속 부분점수로 채점한다."""
    template = JUDGE_SCORE_ABSTAIN if (qtype or "").lower() == "abstention" else JUDGE_SCORE_NORMAL
    prompt = template.format(q=question, gold=gold, ans=answer)
    response = get_client().chat.completions.create(
        model=MODEL_JUDGE,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_score(response.choices[0].message.content or "")
