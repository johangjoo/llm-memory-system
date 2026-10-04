from __future__ import annotations

import json
import secrets
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from openai import OpenAIError
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from server.character_memory import (
    CharacterNotFound,
    UserNotFound,
    get_or_build_character_memory_pack,
    rebuild_all_memory_sections,
    refresh_memory_sections_after_fact_update,
)
from server.database import ensure_schema, get_conn, ping_database, pool
from server.memory import (
    MemoryConfigError,
    build_memory_block,
    extract_session_memories,
    extract_facts_bulk,
    reflect,
    retrieve,
    retrieve_ranked,
    retrieve_reflections,
    store_facts,
)
from server.memory_policy import apply_memory_policy
from server.memory_lifecycle import server_today
from server.future_tasks import FutureTaskResponse, list_future_tasks, store_future_tasks
from server.schemas import (
    ExtractResult,
    FactResponse,
    FactSearchResult,
    CharacterMemoryPackResponse,
    MemoryBlockResponse,
    RecallDetailRequest,
    RecallDetailResponse,
    ReflectionResponse,
    ReflectionSearchResult,
    ReflectRequest,
    SessionCreate,
    SessionResponse,
    UserCreatedResponse,
    UserProfileCreate,
    UserProfileResponse,
)


app = FastAPI(
    title="Mindmate Server",
    description=(
        "MindMate 장기기억 메모리 서버 (PostgreSQL + pgvector).\n\n"
        "주요 사용 흐름:\n"
        "1. `POST /users`로 사용자 프로필 생성\n"
        "2. 응답으로 받은 `user_id` / `api_key`를 클라이언트 또는 Swagger 테스트에 사용\n"
        "3. `POST /users/{user_id}/sessions`로 대화 원본 세션 적재 (raw_sessions)\n"
        "4. `GET /users/{user_id}/sessions` / `GET /users/{user_id}/facts`로 적재·추출 결과 확인\n\n"
        "fact 추출·임베딩·검색·reflection은 이후 단계의 WRITE/READ 파이프라인이 담당합니다."
    ),
    openapi_tags=[
        {"name": "health", "description": "서버와 PostgreSQL 연결 상태를 확인합니다."},
        {"name": "users", "description": "사용자 생성 및 사용자 기본 정보 조회 API입니다."},
        {"name": "sessions", "description": "대화 원본 세션을 적재하고 조회합니다 (raw_sessions)."},
        {"name": "memory", "description": "추출된 장기기억 사실을 조회합니다 (memory_facts)."},
        {"name": "reflection", "description": "누적 facts에서 합성된 상위 통찰(reflections)."},
        {"name": "realtime", "description": "음성 Realtime 세션 instructions에 끼워 넣을 메모리 블록."},
    ],
)
BASE_DIR = Path(__file__).resolve().parent
TEST_USER_FILE = BASE_DIR / "fixtures" / "test_user.json"

USER_COLUMNS = (
    "user_id, user_name, age, birth_date, persona, robot_name, "
    "job, living_info, location, habit, api_key, created_at"
)


@app.on_event("startup")
def startup() -> None:
    ensure_schema()


@app.on_event("shutdown")
def shutdown() -> None:
    if not pool.closed:
        pool.close()


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="사용자 생성 시 발급된 API 키. 우측 상단 Authorize에 한 번 입력하면 보호된 엔드포인트에 모두 적용됩니다.",
)


def generate_api_key() -> str:
    return "mk_" + secrets.token_hex(32)


def require_api_key(
    user_id: str,
    x_api_key: str | None = Security(api_key_header),
) -> None:
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required. Pass X-API-Key header.")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT api_key FROM users WHERE user_id = %s", (user_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if row["api_key"] != x_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.get(
    "/health",
    tags=["health"],
    summary="서버 상태 확인",
    description="FastAPI 서버 프로세스가 살아 있는지만 빠르게 확인합니다. 데이터베이스 연결은 검사하지 않습니다.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/health/db",
    tags=["health"],
    summary="PostgreSQL 연결 확인",
    description="PostgreSQL에 `SELECT 1`을 보내 실제 데이터베이스 연결 가능 여부를 확인합니다.",
)
def health_db() -> dict[str, str]:
    try:
        ping_database()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PostgreSQL connection failed: {exc}") from exc

    return {"status": "ok", "db": "ok"}


@app.post(
    "/users",
    tags=["users"],
    summary="사용자 프로필 생성",
    description=(
        "새 사용자 프로필을 생성합니다. "
        "응답에 포함된 **`user_id`** 값을 `.env`의 `MINDMATE_USER_ID`에 설정하면 "
        "음성 클라이언트가 이 사용자로 동작합니다."
    ),
    response_description="생성된 사용자 프로필 정보 및 API 키 (이후 재조회 불가)",
    response_model=UserCreatedResponse,
)
def create_user_profile(profile: UserProfileCreate) -> dict[str, Any]:
    data = profile.model_dump()
    params = {
        **data,
        "user_id": uuid4().hex,
        "api_key": generate_api_key(),
    }
    with get_conn() as conn:
        row = conn.execute(
            f"""
            INSERT INTO users (user_id, user_name, age, birth_date, persona, robot_name,
                               job, living_info, location, habit, api_key)
            VALUES (%(user_id)s, %(user_name)s, %(age)s, %(birth_date)s, %(persona)s,
                    %(robot_name)s, %(job)s, %(living_info)s, %(location)s, %(habit)s, %(api_key)s)
            RETURNING {USER_COLUMNS}
            """,
            params,
        ).fetchone()
    return serialize_user(row, include_api_key=True)


@app.post(
    "/users/import-test-file",
    tags=["users"],
    summary="테스트 사용자 파일 불러오기",
    description=(
        "`server/fixtures/test_user.json` 파일을 읽어 테스트용 사용자를 생성합니다. "
        "Swagger에서 빠르게 데모 데이터를 만들고 싶을 때 사용합니다."
    ),
    response_description="생성된 테스트 사용자 프로필 정보 및 API 키",
    response_model=UserCreatedResponse,
)
def import_test_user_file() -> dict[str, Any]:
    if not TEST_USER_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Test user file not found: {TEST_USER_FILE}")

    try:
        raw = json.loads(TEST_USER_FILE.read_text(encoding="utf-8"))
        profile = UserProfileCreate.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in test user file: {exc}") from exc

    return create_user_profile(profile)


@app.get(
    "/users",
    tags=["users"],
    summary="사용자 목록 조회",
    description="저장된 모든 사용자 프로필을 최근 생성 순으로 조회합니다.",
    response_description="사용자 프로필 목록",
    response_model=list[UserProfileResponse],
)
def list_user_profiles() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT {USER_COLUMNS} FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [serialize_user(row) for row in rows]


@app.get(
    "/users/{user_id}",
    tags=["users"],
    summary="특정 사용자 조회",
    description=(
        "`user_id`로 특정 사용자 프로필 1건을 조회합니다. "
        "음성 클라이언트 시작 시 사용자 존재 여부를 확인하는 데도 사용됩니다."
    ),
    response_description="사용자 프로필",
    response_model=UserProfileResponse,
)
def get_user_profile(user_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    return serialize_user(require_user_profile(user_id))


@app.post(
    "/users/{user_id}/sessions",
    tags=["sessions"],
    summary="원본 세션 적재",
    description=(
        "대화 한 세션의 turn 목록을 `raw_sessions`에 그대로 적재합니다. "
        "`raw_text`는 `role: text` 줄바꿈 결합으로 서버에서 생성하고, "
        "`transcript`에는 turn 목록을 JSONB로 저장합니다."
    ),
    response_description="저장된 원본 세션",
    response_model=SessionResponse,
)
def create_session(
    user_id: str, payload: SessionCreate, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    require_user_profile(user_id)
    turns = [turn.model_dump() for turn in payload.turns]
    raw_text = "\n".join(f"{turn['role']}: {turn['text']}" for turn in turns)
    session_date = payload.session_date or server_today()
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO raw_sessions (session_id, user_id, character_id, session_date, transcript, raw_text)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING session_id, user_id, character_id, session_date, transcript, raw_text
            """,
            (uuid4().hex, user_id, payload.character_id, session_date, Jsonb(turns), raw_text),
        ).fetchone()
    return serialize_session(row)


@app.get(
    "/users/{user_id}/sessions",
    tags=["sessions"],
    summary="원본 세션 조회",
    description="사용자의 원본 세션을 최근 날짜 순으로 조회합니다.",
    response_description="원본 세션 목록",
    response_model=list[SessionResponse],
)
def list_sessions(user_id: str, _: None = Depends(require_api_key)) -> list[dict[str, Any]]:
    require_user_profile(user_id)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT session_id, user_id, character_id, session_date, transcript, raw_text
            FROM raw_sessions WHERE user_id = %s
            ORDER BY session_date DESC, session_id
            """,
            (user_id,),
        ).fetchall()
    return [serialize_session(row) for row in rows]


@app.get(
    "/users/{user_id}/facts",
    tags=["memory"],
    summary="추출된 사실 조회",
    description=(
        "사용자의 장기기억 사실(`memory_facts`)을 조회합니다. embedding 컬럼은 제외합니다. "
        "기본적으로 만료된 사실(valid_until ≤ 오늘)은 제외하며, `include_expired=true`로 함께 볼 수 있습니다."
    ),
    response_description="추출된 사실 목록",
    response_model=list[FactResponse],
)
def list_facts(
    user_id: str,
    include_expired: bool = Query(
        default=False, description="만료된 사실(valid_until ≤ 오늘)도 포함할지 여부."
    ),
    _: None = Depends(require_api_key),
) -> list[dict[str, Any]]:
    require_user_profile(user_id)
    where = "WHERE user_id = %s"
    if not include_expired:
        where += " AND (valid_until IS NULL OR valid_until > CURRENT_DATE)"
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT memory_id, external_id, user_id, content, summary_for_prompt,
                   memory_type, scope, owner_character_id, domain_tags, importance,
                   event_time, valid_until, validity_status, confidence, sensitivity,
                   conflict_group_id, selection_status, selection_score,
                   selected_count, selection_miss_count, last_selected_at,
                   last_reviewed_at, parked_reason, source_session_id, created_at
            FROM memory_facts {where}
            ORDER BY memory_id
            """,
            (user_id,),
        ).fetchall()
    return [serialize_fact(row) for row in rows]


@app.post(
    "/users/{user_id}/sessions/{session_id}/extract",
    tags=["memory"],
    summary="세션에서 사실 추출·적재",
    description=(
        "지정한 원본 세션의 `raw_text`에서 장기기억 사실을 LLM으로 추출하고, "
        "각 사실에 대해 Mem0식으로 ADD/UPDATE/NOOP/CONFLICT를 판단해 `memory_facts`에 반영합니다. "
        "같은 LLM 요청에서 미래 일정도 추출해 `future_tasks`에 날짜/시/분을 추가 저장합니다. "
        "UPDATE/CONFLICT면 기존 사실의 `valid_until`을 오늘로 만료시키고 새 사실을 추가하며, "
        "NOOP은 스킵합니다. `OPENAI_API_KEY`가 필요합니다."
    ),
    response_description="사실별 처리 결과 + action별 카운트",
    response_model=ExtractResult,
)
def extract_session(
    user_id: str, session_id: str, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    require_user_profile(user_id)
    with get_conn() as conn:
        session = conn.execute(
            """
            SELECT session_id, character_id, session_date, raw_text, transcript
            FROM raw_sessions WHERE session_id = %s AND user_id = %s
            """,
            (session_id, user_id),
        ).fetchone()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    try:
        facts, task_candidates = extract_session_memories(
            session["transcript"] or [],
            session["session_date"],
            active_character_id=session.get("character_id"),
        )
        with get_conn() as conn:
            # Serialize retries for this session so the task upserts share one transaction.
            conn.execute("SELECT session_id FROM raw_sessions WHERE session_id = %s AND user_id = %s FOR UPDATE",
                         (session_id, user_id))
            results = store_facts(
                conn,
                user_id,
                session_id,
                session["session_date"],
                facts,
                active_character_id=session.get("character_id"),
            )
            future_result = store_future_tasks(
                conn, user_id, session_id, session.get("character_id"),
                task_candidates, session["transcript"] or [],
            )
            updated_memory_sections = refresh_memory_sections_after_fact_update(
                conn,
                user_id=user_id,
                results=results,
                active_character_id=session.get("character_id"),
                session_id=session_id,
            )
    except MemoryConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Invalid memory/task extraction response.") from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {exc}") from exc

    counts = {"ADD": 0, "UPDATE": 0, "NOOP": 0, "CONFLICT": 0}
    decisions: list[dict[str, Any]] = []
    for result in results:
        counts[result["action"]] = counts.get(result["action"], 0) + 1
        decisions.append({
            "action": result["action"],
            "content": result["content"],
            "fact": serialize_fact(result["fact"]) if result["fact"] else None,
            "expired_id": result["expired_id"],
            "reason": result["reason"],
        })
    return {
        "decisions": decisions,
        "counts": counts,
        "future_tasks": future_result["tasks"],
        "future_tasks_skipped": future_result["skipped"],
        "updated_memory_sections": updated_memory_sections,
        "updated_prompt_caches": updated_memory_sections,
    }


@app.get(
    "/users/{user_id}/future-tasks",
    tags=["memory"], summary="미래 일정/알람 후보 조회", response_model=list[FutureTaskResponse],
)
def get_future_tasks(
    user_id: str,
    task_date: date | None = None,
    ready_only: bool = False,
    character_id: Literal["study", "cooking"] | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    _: None = Depends(require_api_key),
) -> list[dict[str, Any]]:
    """pending 일정을 반환한다. ready_only는 날짜와 시각이 확정된 행만 선택한다.

    알람 실행기는 scheduled_at을 기준으로 별도로 만기 판단해야 한다.
    미처리된 지난 일정도 반환하여 재시작 후 누락 여부를 판단할 수 있게 한다.
    """
    with get_conn() as conn:
        return list_future_tasks(conn, user_id, task_date=task_date, ready_only=ready_only,
                                 character_id=character_id, limit=limit)


@app.post(
    "/users/{user_id}/extract-all",
    tags=["memory"],
    summary="전체 세션 일괄 추출 (저비용)",
    description=(
        "그 사용자의 모든 raw_sessions를 **한 번의 LLM 호출**로 통째로 추출해 memory_facts에 "
        "적재합니다. 세션마다 호출하는 `/extract`(세션당 1회 + 중복판단)보다 OpenAI 비용이 훨씬 "
        "적습니다. 시간 흐름에 따라 바뀐 정보는 LLM이 최신값만 남깁니다(세션 단위 ADD/UPDATE 판단은 "
        "생략). 기본적으로 기존 facts를 비우고 새로 적재합니다(`reset=false`로 누적 가능). "
        "`OPENAI_API_KEY`가 필요합니다."
    ),
    response_description="적재된 fact 수",
    response_model=ExtractResult,
)
def extract_all_sessions(
    user_id: str,
    reset: bool = Query(default=True, description="True면 기존 facts/reflections를 비우고 새로 적재."),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    require_user_profile(user_id)
    with get_conn() as conn:
        sessions = conn.execute(
            """
            SELECT session_id, character_id, session_date, raw_text FROM raw_sessions
            WHERE user_id = %s ORDER BY session_date, session_id
            """,
            (user_id,),
        ).fetchall()
    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions to extract.")
    try:
        facts = extract_facts_bulk(
            [
                {
                    "session_date": s["session_date"],
                    "character_id": s.get("character_id"),
                    "raw_text": s["raw_text"],
                }
                for s in sessions
            ]
        )
        with get_conn() as conn:
            if reset:
                conn.execute("DELETE FROM memory_facts WHERE user_id = %s", (user_id,))
                conn.execute("DELETE FROM reflections WHERE user_id = %s", (user_id,))
            # 일괄 모드는 중복판단 없이 그대로 ADD (LLM이 이미 전체를 보고 정리함)
            results = store_facts(
                conn, user_id, sessions[0]["session_id"], None, facts, enable_update=False
            )
            updated_memory_sections = rebuild_all_memory_sections(
                conn,
                user_id=user_id,
                session_id=sessions[-1]["session_id"],
            )
    except MemoryConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {exc}") from exc

    counts = {"ADD": len(results), "UPDATE": 0, "NOOP": 0, "CONFLICT": 0}
    decisions = [
        {
            "action": "ADD",
            "content": r["content"],
            "fact": serialize_fact(r["fact"]) if r["fact"] else None,
            "expired_id": None,
            "reason": "bulk extract",
        }
        for r in results
    ]
    return {
        "decisions": decisions,
        "counts": counts,
        "updated_memory_sections": updated_memory_sections,
        "updated_prompt_caches": updated_memory_sections,
    }


@app.get(
    "/users/{user_id}/facts/search",
    tags=["memory"],
    summary="사실 검색",
    description=(
        "질문 `q`로 사실을 검색합니다. `OPENAI_API_KEY`가 필요합니다.\n\n"
        "- `ranking=weighted`(기본): Generative Agents식 relevance·recency·importance "
        "정규화 가중합. 후보 `pool`개를 먼저 가져와 재정렬합니다.\n"
        "- `ranking=relevance`: 순수 코사인 유사도."
    ),
    response_description="상위 사실 목록(점수 포함)",
    response_model=list[FactSearchResult],
)
def search_facts(
    user_id: str,
    q: str = Query(description="검색할 질문/문장."),
    k: int = Query(default=5, ge=1, le=50, description="반환할 상위 결과 수."),
    ranking: Literal["weighted", "relevance"] = Query(
        default="weighted", description="랭킹 방식. weighted=가중합 / relevance=코사인."
    ),
    pool: int = Query(
        default=20, ge=1, le=200, description="weighted 모드에서 재정렬할 후보 수(>= k 권장)."
    ),
    include_prompt_excluded: bool = Query(
        default=False,
        description=(
            "True면 prompt pack에서 제외된 parked/archived/sensitive 기억도 검색 후보에 포함합니다. "
            "scope=blocked와 outdated fact는 계속 제외합니다."
        ),
    ),
    _: None = Depends(require_api_key),
) -> list[dict[str, Any]]:
    require_user_profile(user_id)
    try:
        with get_conn() as conn:
            if ranking == "weighted":
                rows = retrieve_ranked(
                    conn,
                    user_id,
                    q,
                    k,
                    pool,
                    include_prompt_excluded=include_prompt_excluded,
                )
            else:
                rows = retrieve(
                    conn,
                    user_id,
                    q,
                    k,
                    include_prompt_excluded=include_prompt_excluded,
                )
    except MemoryConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {exc}") from exc
    return [serialize_fact(row, include_score=True) for row in rows]


@app.post(
    "/users/{user_id}/recall-detail",
    tags=["memory"],
    summary="MAP 필터를 적용한 2단계 상세 기억 검색",
    description=(
        "Realtime function calling에서 쓰는 저비용 2단계 recall 경로입니다. "
        "vector 후보를 먼저 가져온 뒤 active character 기준 MAP을 적용해 다른 character private memory를 제외합니다."
    ),
    response_description="MAP을 통과한 상세 기억 검색 결과",
    response_model=RecallDetailResponse,
)
def recall_detail(
    user_id: str,
    payload: RecallDetailRequest,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    require_user_profile(user_id)
    with get_conn() as conn:
        character = conn.execute(
            """
            SELECT character_id, name, description, allowed_domains, blocked_domains, tone
            FROM characters WHERE character_id = %s
            """,
            (payload.active_character_id,),
        ).fetchone()
        if character is None:
            raise HTTPException(
                status_code=404,
                detail=f"Character not found: {payload.active_character_id}",
            )
        try:
            rows = retrieve_ranked(
                conn,
                user_id,
                payload.query,
                payload.pool,
                payload.pool,
                include_prompt_excluded=True,
            )
        except MemoryConfigError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OpenAIError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {exc}") from exc

        allowed = apply_memory_policy(
            conn,
            user_id=user_id,
            active_character_id=payload.active_character_id,
            memories=rows,
            character=character,
            query=payload.query,
            log_decisions=payload.log_decisions,
            allow_sensitive=True,
        )

    returned = allowed[: payload.k]
    results: list[dict[str, Any]] = []
    for row in returned:
        item = serialize_fact(row, include_score=True)
        item["policy_score"] = float(row.get("policy_score") or 0.0)
        item["policy_reason"] = row.get("policy_reason")
        results.append(item)
    return {
        "query": payload.query,
        "active_character_id": payload.active_character_id,
        "results": results,
        "counts": {
            "retrieved": len(rows),
            "allowed": len(allowed),
            "returned": len(returned),
            "blocked": len(rows) - len(allowed),
        },
    }


@app.post(
    "/users/{user_id}/reflect",
    tags=["reflection"],
    summary="누적 facts에서 reflection 합성",
    description=(
        "Generative Agents식으로 사용자의 누적 활성 facts에서 상위 통찰(reflection)을 LLM이 합성하고 "
        "`reflections` 테이블에 적재합니다. 각 reflection은 임베딩되고 근거 fact id 목록을 포함합니다. "
        "후보가 2개 미만이면 빈 결과를 반환합니다. `OPENAI_API_KEY`가 필요합니다."
    ),
    response_description="새로 적재된 reflection 목록",
    response_model=list[ReflectionResponse],
)
def reflect_endpoint(
    user_id: str,
    payload: ReflectRequest | None = None,
    _: None = Depends(require_api_key),
) -> list[dict[str, Any]]:
    require_user_profile(user_id)
    body = payload or ReflectRequest()
    try:
        with get_conn() as conn:
            inserted = reflect(
                conn,
                user_id,
                limit=body.limit,
                since=body.since,
                max_reflections=body.max_reflections,
            )
    except MemoryConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {exc}") from exc
    return [serialize_reflection(row) for row in inserted]


@app.get(
    "/users/{user_id}/reflections",
    tags=["reflection"],
    summary="저장된 reflection 목록",
    description="사용자에게 누적된 reflection을 최신순으로 조회합니다.",
    response_description="reflection 목록",
    response_model=list[ReflectionResponse],
)
def list_reflections(
    user_id: str, _: None = Depends(require_api_key)
) -> list[dict[str, Any]]:
    require_user_profile(user_id)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT reflection_id, user_id, summary, source_memory_ids, created_at
            FROM reflections WHERE user_id = %s
            ORDER BY reflection_id DESC
            """,
            (user_id,),
        ).fetchall()
    return [serialize_reflection(row) for row in rows]


@app.get(
    "/users/{user_id}/reflections/search",
    tags=["reflection"],
    summary="reflection 벡터 검색",
    description=(
        "질문 `q`를 임베딩해 reflections를 코사인 유사도 기준 상위 `k`개 검색합니다. "
        "`OPENAI_API_KEY`가 필요합니다."
    ),
    response_description="유사도 상위 reflection 목록(score 포함)",
    response_model=list[ReflectionSearchResult],
)
def search_reflections(
    user_id: str,
    q: str = Query(description="검색할 질문/문장."),
    k: int = Query(default=5, ge=1, le=50, description="반환할 상위 결과 수."),
    _: None = Depends(require_api_key),
) -> list[dict[str, Any]]:
    require_user_profile(user_id)
    try:
        with get_conn() as conn:
            rows = retrieve_reflections(conn, user_id, q, k)
    except MemoryConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {exc}") from exc
    return [serialize_reflection(row, include_score=True) for row in rows]


@app.get(
    "/users/{user_id}/characters/{character_id}/memory-pack",
    tags=["realtime"],
    summary="Active character용 Realtime memory pack 생성",
    description=(
        "study/cooking 같은 active character 기준으로 shared memory와 해당 character private memory만 "
        "선택해 Realtime instructions에 붙일 compact memory pack을 만듭니다. 다른 character private memory, "
        "blocked/outdated/sensitive memory는 제외됩니다."
    ),
    response_description="캐릭터별 memory pack + MAP 필터링 count",
    response_model=CharacterMemoryPackResponse,
)
def get_character_memory_pack(
    user_id: str,
    character_id: str,
    shared_limit: int = Query(default=0, ge=0, le=100, description="포함할 shared memory 최대 개수. 0이면 개수 제한 없이 점수 순서와 문자 예산으로 선택합니다."),
    private_limit: int = Query(default=0, ge=0, le=100, description="포함할 character_private memory 최대 개수. 0이면 개수 제한 없이 점수 순서와 문자 예산으로 선택합니다."),
    reflection_limit: int = Query(default=0, ge=0, le=50, description="포함할 reflection 최대 개수. 0이면 내부 상한까지 포함하고 전체 문자 예산으로 제한합니다."),
    max_prompt_chars: int = Query(default=12000, ge=1000, le=30000, description="memory pack 최대 문자 수."),
    refresh_cache: bool = Query(default=False, description="True면 캐시를 무시하고 다시 조립해 저장."),
    log_decisions: bool = Query(default=False, description="True면 MAP allow/block 결정을 policy_decisions에 기록."),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    require_user_profile(user_id)
    try:
        with get_conn() as conn:
            return get_or_build_character_memory_pack(
                conn,
                user_id=user_id,
                character_id=character_id,
                shared_limit=shared_limit,
                private_limit=private_limit,
                reflection_limit=reflection_limit,
                max_prompt_chars=max_prompt_chars,
                refresh_cache=refresh_cache,
                log_decisions=log_decisions,
            )
    except CharacterNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Character not found: {character_id}") from exc
    except UserNotFound as exc:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}") from exc


@app.get(
    "/users/{user_id}/memory-block",
    tags=["realtime"],
    summary="Realtime instructions용 메모리 블록 생성",
    description=(
        "OpenAI Realtime 프롬프팅 가이드의 Context 권장 구조 "
        "(Current Insights / Recent Memory / Background / User Profile)로 조립된 "
        "한국어 메모리 블록 텍스트를 반환합니다. 클라이언트는 받은 `text`를 기존 "
        "instructions 뒤에 그대로 붙여 Realtime 세션을 시작하면 됩니다.\n\n"
        "- 세션 시작 시점에는 사용자 질문이 없어 코사인 relevance를 쓰지 않고, "
        "recency·importance만 후보풀에서 min-max 정규화 후 가중합으로 top-`m_facts`를 뽑습니다.\n"
        "- 그 안에서 recency 정규값 ≥ `recent_split`은 Recent, 미만은 Background로 분리.\n"
        "- 만료된 사실(`valid_until` ≤ 오늘)은 이 경로에서도 제외합니다."
    ),
    response_description="조립된 메모리 블록 + 섹션별 카운트",
    response_model=MemoryBlockResponse,
)
def get_memory_block(
    user_id: str,
    k_reflections: int = Query(default=5, ge=0, le=20, description="포함할 reflection 최대 수."),
    m_facts: int = Query(default=0, ge=0, le=500, description="포함할 fact 최대 수. 0이면 제한 없음(전부)."),
    recent_split: float = Query(
        default=0.5, ge=0.0, le=1.0,
        description="recency 정규값이 이 값 이상이면 Recent로, 미만이면 Background로 분류.",
    ),
    consolidate: bool = Query(
        default=False,
        description="True면 facts를 LLM으로 정리(중복→최신, 습관→일반화, 모순→최신)한 뒤 조립. LLM 1회 호출.",
    ),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    require_user_profile(user_id)
    with get_conn() as conn:
        return build_memory_block(
            conn, user_id,
            k_reflections=k_reflections, m_facts=(m_facts or None),
            recent_split=recent_split, consolidate=consolidate,
        )


def require_user_profile(user_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT {USER_COLUMNS} FROM users WHERE user_id = %s", (user_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return row


def _iso(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def serialize_user(row: dict[str, Any], include_api_key: bool = False) -> dict[str, Any]:
    result = {
        "user_id": row["user_id"],
        "user_name": row["user_name"],
        "age": row["age"],
        "birth_date": _iso(row["birth_date"]),
        "persona": row["persona"],
        "robot_name": row["robot_name"],
        "job": row["job"],
        "living_info": row["living_info"],
        "location": row["location"],
        "habit": row["habit"],
        "created_at": _iso(row["created_at"]),
    }
    if include_api_key:
        result["api_key"] = row["api_key"]
    return result


def serialize_session(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "user_id": row["user_id"],
        "character_id": row.get("character_id"),
        "session_date": _iso(row["session_date"]),
        "transcript": row["transcript"] or [],
        "raw_text": row["raw_text"] or "",
    }


def serialize_reflection(row: dict[str, Any], include_score: bool = False) -> dict[str, Any]:
    result = {
        "reflection_id": row["reflection_id"],
        "user_id": row["user_id"],
        "character_id": row.get("character_id"),
        "summary": row["summary"],
        "source_memory_ids": list(row["source_memory_ids"] or []),
        "created_at": _iso(row["created_at"]),
    }
    if include_score:
        result["score"] = float(row["score"])
    return result


def serialize_fact(row: dict[str, Any], include_score: bool = False) -> dict[str, Any]:
    result = {
        "memory_id": row["memory_id"],
        "external_id": row.get("external_id"),
        "user_id": row["user_id"],
        "content": row["content"],
        "summary_for_prompt": row.get("summary_for_prompt"),
        "memory_type": row["memory_type"],
        "scope": row.get("scope", "shared"),
        "owner_character_id": row.get("owner_character_id"),
        "domain_tags": row["domain_tags"],
        "importance": row["importance"],
        "event_time": _iso(row["event_time"]),
        "valid_until": _iso(row["valid_until"]),
        "validity_status": row.get("validity_status", "current"),
        "confidence": row.get("confidence"),
        "sensitivity": row.get("sensitivity", "normal"),
        "conflict_group_id": row.get("conflict_group_id"),
        "selection_status": row.get("selection_status", "candidate"),
        "selection_score": row.get("selection_score"),
        "selected_count": row.get("selected_count", 0),
        "selection_miss_count": row.get("selection_miss_count", 0),
        "last_selected_at": _iso(row.get("last_selected_at")),
        "last_reviewed_at": _iso(row.get("last_reviewed_at")),
        "parked_reason": row.get("parked_reason"),
        "source_session_id": row["source_session_id"],
        "created_at": _iso(row["created_at"]),
    }
    if include_score:
        result["score"] = float(row["score"])
        for component in ("relevance", "recency", "importance_score", "lexical"):
            if component in row:
                result[component] = float(row[component])
    return result
