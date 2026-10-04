"""데이터셋 로더.

내부 포맷:
{
  "question_id": str,
  "question_type": str,            # LongMemEval 원본 축 그대로
  "question": str,
  "answer": str,                   # gold
  "sessions": [
      {"date": str | None, "turns": [{"role": str, "text": str}, ...]},
      ...
  ],
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LME_DIR = DATA_DIR / "longmemeval"


def _normalize_turn(raw: dict[str, Any]) -> dict[str, str]:
    """LongMemEval은 turn에 `content` 필드를 쓴다(매뉴얼 가정과 다름)."""
    role = raw.get("role") or "user"
    text = raw.get("text") or raw.get("content") or ""
    return {"role": role, "text": text}


def _load_longmemeval_file(path: Path, limit: int | None) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for entry in raw[: limit] if limit else raw:
        sessions_raw: Iterable[Any] = entry.get("haystack_sessions") or []
        dates = entry.get("haystack_dates") or []
        sessions: list[dict[str, Any]] = []
        for i, session in enumerate(sessions_raw):
            # session 자체가 turn의 리스트인 경우(현 LongMemEval-cleaned)와
            # {"turns": [...], "date": ...} 형태 양쪽을 지원
            if isinstance(session, list):
                turns = [_normalize_turn(t) for t in session if isinstance(t, dict)]
                date = dates[i] if i < len(dates) else None
            elif isinstance(session, dict):
                turns = [_normalize_turn(t) for t in session.get("turns", []) if isinstance(t, dict)]
                date = session.get("date") or (dates[i] if i < len(dates) else None)
            else:
                continue
            if not turns:
                continue
            sessions.append({"date": date, "turns": turns})

        items.append({
            "question_id": entry["question_id"],
            "question_type": entry.get("question_type", "unknown"),
            "question": entry.get("question", ""),
            "answer": entry.get("answer", ""),
            "sessions": sessions,
        })
    return items


def load_longmemeval_s(limit: int | None = None) -> list[dict[str, Any]]:
    """LongMemEval-S(짧은 haystack subset, 500문항)."""
    return _load_longmemeval_file(LME_DIR / "longmemeval_s_cleaned.json", limit)


def load_longmemeval_m(limit: int | None = None) -> list[dict[str, Any]]:
    """LongMemEval-M(긴 haystack). 메모리·시간 많이 잡아먹으니 limit 권장."""
    return _load_longmemeval_file(LME_DIR / "longmemeval_m_cleaned.json", limit)


def load_longmemeval_oracle(limit: int | None = None) -> list[dict[str, Any]]:
    """oracle: 정답에 기여한 세션만 포함된 버전. 디버깅·상한 측정용."""
    return _load_longmemeval_file(LME_DIR / "longmemeval_oracle.json", limit)


def load_locomo(limit: int | None = None) -> list[dict[str, Any]]:
    """LoCoMo10. 구조가 LongMemEval과 다름 — 1차 실험에선 사용하지 않음(stub)."""
    raise NotImplementedError(
        "LoCoMo 로더는 2차에서 작성 — 원본 구조(qa/conversation/event_summary 등)에 맞춰 별도 매핑 필요."
    )
