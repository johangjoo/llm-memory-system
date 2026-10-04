"""Future plans extracted alongside facts; this module does not deliver alarms."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from server.memory_lifecycle import SERVER_TIMEZONE


FUTURE_TASK_RULES = """
추가로 알람 준비용 future_tasks 배열을 같은 JSON 객체에 출력한다.
최종 출력: {"facts": [...], "future_tasks": [...]}.
facts 규칙은 그대로 유지하고, 장기기억에서는 제외한 일회성 미래 계획도 future_tasks에는 남긴다.
- 사용자 자신이 하기로 한 미래 할 일, 약속, 마감, 알림 요청만 추출한다.
  AI의 제안만 있는 경우, 가정/질문, 과거 완료, 부정/취소된 계획은 제외한다.
  같은 대화에서 수정한 일정은 최종 내용 하나만 남긴다. 단순 희망/일반 습관은 제외한다.
- 대화 날짜 기준으로 오늘/내일/모레/다음 주/명시한 연월일을 해석한다.
  정확한 날짜를 확정할 수 없으면 due_date=null. 임의로 날짜나 연도를 만들어내지 않는다.
- 명시한 시/분은 24시간 HH:MM으로 보존한다. 오후 3시 30분=15:30/minute,
  오전 9시=09:00/hour, 오후 3시 반=15:30/minute. 자정=00:00, 정오=12:00.
  시각을 말하지 않았거나 '오후쯤', 오전/오후가 불명확한 '3시'이면 due_time=null.
  날짜만 알면 time_precision=date, 날짜도 없으면 unspecified.
  발화 시각이 제공되지 않으므로 '10분 뒤'를 요청 처리 시각으로 계산하지 말고 미정으로 남긴다.
- title은 일정 제목이다. 한 발화에 여러 일정이 있으면 각각 분리한다.
- source_turn_index는 아래 번호가 붙은 원 대화의 0부터 시작하는 인덱스다.
  source_quote는 해당 user 발화에서 일정을 증명하는 연속된 원문을 그대로 복사한다.
  한 발화에서 여러 일정을 분리할 때 각 일정에 해당하는 서로 다른 구절을 사용한다.
- 일정이 없으면 future_tasks=[]를 출력한다.
각 future_tasks 항목:
{"title":"발표 준비", "due_date":"2026-09-28 또는 null", "due_time":"15:30 또는 null",
 "time_precision":"unspecified|date|hour|minute", "source_turn_index":0,
 "source_quote":"내일 오후 3시 30분에 발표 준비할 거야"}
"""


class ExtractedTask(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=500)
    due_date: date | None = None
    due_time: time | None = None
    time_precision: Literal["unspecified", "date", "hour", "minute"]
    source_turn_index: int = Field(ge=0, strict=True)
    source_quote: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_precision(self):
        if self.due_time is not None:
            if self.due_time.tzinfo or self.due_time.second or self.due_time.microsecond:
                raise ValueError("Use local time at minute precision")
            if self.time_precision not in {"hour", "minute"}:
                raise ValueError("Time requires hour/minute precision")
            if self.time_precision == "hour" and self.due_time.minute:
                raise ValueError("Hour precision requires minute zero")
        elif self.time_precision != ("date" if self.due_date else "unspecified"):
            raise ValueError("Missing time must not imply an exact alarm")
        return self


class FutureTaskResponse(ExtractedTask):
    task_id: int
    user_id: str
    source_session_id: str
    character_id: str | None
    timezone: str
    scheduled_at: datetime | None
    needs_clarification: bool
    status: Literal["pending", "completed", "cancelled"]
    created_at: datetime
    updated_at: datetime


def normalize_task(candidate: Any, turns: list[dict], *, now: datetime | None = None) -> dict:
    task = ExtractedTask.model_validate(candidate)
    if task.source_turn_index >= len(turns):
        raise ValueError("Invalid source turn")
    turn = turns[task.source_turn_index]
    if turn.get("role") != "user" or task.source_quote not in turn.get("text", ""):
        raise ValueError("Task must cite an original user utterance")
    tz = ZoneInfo(SERVER_TIMEZONE)
    now = (now or datetime.now(timezone.utc)).astimezone(tz)
    scheduled_at = None
    if task.due_date and task.due_time is not None:
        local = datetime.combine(task.due_date, task.due_time)
        # Avoid silently choosing an ambiguous/nonexistent DST wall time.
        first, second = local.replace(tzinfo=tz, fold=0), local.replace(tzinfo=tz, fold=1)
        if first.utcoffset() != second.utcoffset():
            raise ValueError("Ambiguous or nonexistent local time")
        scheduled_at = first
    if task.due_date and task.due_date < now.date():
        raise ValueError("Past task")
    if scheduled_at and scheduled_at <= now:
        raise ValueError("Past task time")
    result = task.model_dump()
    result.update(timezone=SERVER_TIMEZONE, scheduled_at=scheduled_at,
                  needs_clarification=scheduled_at is None)
    # Evidence + occurrence, rather than an LLM-rephrased title, identifies a retry.
    # One utterance can describe multiple occurrences on different days/times.
    result["source_key"] = hashlib.sha256(
        f"{task.source_turn_index}:{task.source_quote}:{task.due_date}:{task.due_time}".encode("utf-8")
    ).hexdigest()
    return result


def store_future_tasks(conn, user_id: str, session_id: str, character_id: str | None,
                       candidates: list, turns: list[dict], *, now: datetime | None = None) -> dict:
    saved, skipped = [], 0
    for candidate in candidates:
        try:
            task = normalize_task(candidate, turns, now=now)
        except (ValidationError, ValueError, TypeError):
            skipped += 1
            continue
        row = conn.execute(
            """INSERT INTO future_tasks
                (user_id, source_session_id, character_id, title, due_date, due_time,
                 timezone, time_precision, scheduled_at, needs_clarification,
                 source_turn_index, source_quote, source_key)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (user_id, source_session_id, source_key) DO UPDATE SET
                 title=EXCLUDED.title, due_date=EXCLUDED.due_date, due_time=EXCLUDED.due_time,
                 timezone=EXCLUDED.timezone, time_precision=EXCLUDED.time_precision,
                 scheduled_at=EXCLUDED.scheduled_at, needs_clarification=EXCLUDED.needs_clarification,
                 updated_at=NOW()
               RETURNING *""",
            (user_id, session_id, character_id, task["title"], task["due_date"], task["due_time"],
             task["timezone"], task["time_precision"], task["scheduled_at"],
             task["needs_clarification"], task["source_turn_index"], task["source_quote"], task["source_key"]),
        ).fetchone()
        saved.append(row)
    return {"tasks": saved, "skipped": skipped}


def list_future_tasks(conn, user_id: str, *, task_date: date | None = None,
                      ready_only: bool = False, character_id: str | None = None,
                      limit: int = 100) -> list[dict]:
    return conn.execute(
        """SELECT * FROM future_tasks WHERE user_id = %s AND status = 'pending'
           AND (%s::date IS NULL OR due_date = %s)
           AND (NOT %s OR scheduled_at IS NOT NULL)
           AND (%s::text IS NULL OR character_id IS NULL OR character_id = %s)
           ORDER BY due_date NULLS LAST, due_time NULLS LAST, task_id LIMIT %s""",
        (user_id, task_date, task_date, ready_only, character_id, character_id, limit),
    ).fetchall()
