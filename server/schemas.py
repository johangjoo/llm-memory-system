"""Pydantic 요청·응답 스키마.

Swagger UI에서 각 필드 설명과 예시를 한국어로 표시하고,
응답 구조도 명시하여 API 문서를 완성합니다.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field
from server.future_tasks import FutureTaskResponse


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 요청 스키마 (Request)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UserProfileCreate(BaseModel):
    """사용자 프로필 생성 요청."""

    user_name: str = Field(
        description="사용자 실명 또는 닉네임. (`user_name`)",
        examples=["김민수"],
    )
    age: int | None = Field(
        default=None,
        ge=0,
        description="사용자 나이. 선택 사항. (`age`)",
        examples=[23],
    )
    birth_date: date | None = Field(
        default=None,
        description="생년월일. YYYY-MM-DD 형식. 선택 사항. (`birth_date`)",
        examples=["2002-03-15"],
    )
    persona: str | None = Field(
        default=None,
        description=(
            "AI 어시스턴트의 말투·성격 지시문. 세션 프롬프트에 그대로 삽입됩니다. "
            "선택 사항. (`persona`)"
        ),
        examples=["친한 친구처럼 반말로 편하게 대화해줘."],
    )
    robot_name: str | None = Field(
        default=None,
        description="사용자가 로봇을 부르는 이름. 세션 프롬프트에 사용됩니다. 선택 사항. (`robot_name`)",
        examples=["진석"],
    )
    job: str | None = Field(
        default=None,
        description="직업 또는 역할. 맥락 파악에 사용됩니다. 선택 사항. (`job`)",
        examples=["대학생"],
    )
    living_info: str | None = Field(
        default=None,
        description="거주 환경 요약. 선택 사항. (`living_info`)",
        examples=["가족과 함께 거주 중인 4학년 대학생."],
    )
    location: str | None = Field(
        default=None,
        description="주요 거주 지역 또는 도시명. 선택 사항. (`location`)",
        examples=["경기도 안산시"],
    )
    habit: str | None = Field(
        default=None,
        description="평소 취미·생활 습관 요약. 선택 사항. (`habit`)",
        examples=["주말에 등산하거나 유튜브로 요리 영상 보는 걸 좋아함."],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_name": "김민수",
                "age": 23,
                "birth_date": "2002-03-15",
                "persona": "친한 친구처럼 반말로 편하게 대화해줘.",
                "robot_name": "진석",
                "job": "대학생",
                "living_info": "가족과 함께 거주 중인 4학년 대학생.",
                "location": "경기도 안산시",
                "habit": "주말에 등산하거나 유튜브로 요리 영상 보는 걸 좋아함.",
            }
        }
    }


class Turn(BaseModel):
    """대화 한 턴."""

    role: Literal["user", "assistant"] = Field(
        description="발화 주체. `user` 또는 `assistant`. (`role`)",
        examples=["user"],
    )
    text: str = Field(
        description="발화 내용. (`text`)",
        examples=["오늘 강화학습 공부했어."],
    )


class SessionCreate(BaseModel):
    """원본 세션(raw_sessions) 저장 요청.

    음성/텍스트 대화 한 세션의 turn 목록을 그대로 적재합니다.
    fact 추출·임베딩은 이후 단계의 WRITE 파이프라인이 담당합니다.
    """

    session_date: date | None = Field(
        default=None,
        description=(
            "세션이 진행된 날짜. YYYY-MM-DD 형식. "
            "생략하면 요청 시각 기준 오늘 날짜로 저장됩니다. (`session_date`)"
        ),
        examples=["2026-05-26"],
    )
    character_id: str | None = Field(
        default=None,
        description=(
            "이 세션의 active character ID. 예: `study`, `cooking`. "
            "없으면 캐릭터 미지정 세션으로 저장됩니다. (`character_id`)"
        ),
        examples=["study"],
    )
    turns: list[Turn] = Field(
        description="대화 turn 목록. 최소 1개. (`turns`)",
        min_length=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_date": "2026-05-26",
                "character_id": "study",
                "turns": [
                    {"role": "user", "text": "오늘은 TD learning 공부했어. bootstrapping이 핵심이더라"},
                    {"role": "assistant", "text": "맞아요. 다음 추정값으로 현재 추정값을 업데이트하는 게 TD의 핵심입니다."},
                ],
            }
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 응답 스키마 (Response)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UserProfileResponse(BaseModel):
    """사용자 프로필 응답."""

    user_id: str = Field(description="사용자 고유 ID. `.env`의 `MINDMATE_USER_ID`에 설정하는 값. (`user_id`)")
    user_name: str = Field(description="사용자 실명 또는 닉네임. (`user_name`)")
    age: int | None = Field(default=None, description="나이. (`age`)")
    birth_date: str | None = Field(default=None, description="생년월일. YYYY-MM-DD. (`birth_date`)")
    persona: str | None = Field(default=None, description="AI 말투·성격 지시문. (`persona`)")
    robot_name: str | None = Field(default=None, description="로봇 호출명. (`robot_name`)")
    job: str | None = Field(default=None, description="직업. (`job`)")
    living_info: str | None = Field(default=None, description="거주 환경. (`living_info`)")
    location: str | None = Field(default=None, description="거주 지역. (`location`)")
    habit: str | None = Field(default=None, description="취미·생활 습관. (`habit`)")
    created_at: str = Field(description="생성 시각. ISO 8601. (`created_at`)")


class UserCreatedResponse(UserProfileResponse):
    """사용자 생성 직후 응답. API 키를 한 번만 포함합니다."""

    api_key: str = Field(
        description=(
            "발급된 API 키. 생성 시 **한 번만** 반환됩니다. "
            "`.env`의 `MINDMATE_API_KEY`에 저장하세요. (`api_key`)"
        )
    )


class SessionResponse(BaseModel):
    """원본 세션 응답."""

    session_id: str = Field(description="세션 고유 ID. (`session_id`)")
    user_id: str = Field(description="사용자 ID. (`user_id`)")
    character_id: str | None = Field(default=None, description="세션의 active character ID. (`character_id`)")
    session_date: str | None = Field(default=None, description="세션 날짜. YYYY-MM-DD. (`session_date`)")
    transcript: list[dict[str, Any]] = Field(description="대화 turn 목록(JSONB 원본). (`transcript`)")
    raw_text: str = Field(description="turn을 `role: text` 줄바꿈으로 결합한 원문. (`raw_text`)")


class FactResponse(BaseModel):
    """추출된 사실(memory_facts) 응답. embedding은 제외합니다."""

    memory_id: int = Field(description="사실 고유 ID. (`memory_id`)")
    external_id: str | None = Field(default=None, description="시드/평가용 외부 ID. (`external_id`)")
    user_id: str = Field(description="사용자 ID. (`user_id`)")
    content: str = Field(description="문장 1개로 정리된 사실. (`content`)")
    summary_for_prompt: str | None = Field(default=None, description="프롬프트 삽입용 짧은 요약. (`summary_for_prompt`)")
    memory_type: str | None = Field(default=None, description="사실 유형. (`memory_type`)")
    scope: Literal["shared", "character_private", "blocked"] = Field(
        default="shared",
        description="메모리 접근 범위. (`scope`)",
    )
    owner_character_id: str | None = Field(default=None, description="character_private memory의 소유 캐릭터. (`owner_character_id`)")
    domain_tags: list[str] | None = Field(default=None, description="도메인 태그. (`domain_tags`)")
    importance: int | None = Field(default=None, description="중요도 1~5. (`importance`)")
    event_time: str | None = Field(default=None, description="사실의 발생 시점. YYYY-MM-DD. (`event_time`)")
    valid_until: str | None = Field(default=None, description="유효 만료일(망각축). YYYY-MM-DD. (`valid_until`)")
    validity_status: Literal["current", "outdated", "uncertain"] = Field(
        default="current",
        description="현재/오래됨/불확실 상태. (`validity_status`)",
    )
    confidence: float | None = Field(default=None, description="추출 신뢰도 0.0~1.0. (`confidence`)")
    sensitivity: Literal["low", "normal", "sensitive"] = Field(
        default="normal",
        description="민감도. (`sensitivity`)",
    )
    conflict_group_id: str | None = Field(default=None, description="갱신/충돌 그룹 ID. (`conflict_group_id`)")
    selection_status: Literal["candidate", "selected", "parked", "archived", "blocked"] = Field(
        default="candidate",
        description="Prompt selection status. (`selection_status`)",
    )
    selection_score: float | None = Field(default=None, description="Last prompt selection score. (`selection_score`)")
    selected_count: int = Field(default=0, description="Prompt section selected count. (`selected_count`)")
    selection_miss_count: int = Field(default=0, description="Consecutive section-selection misses. (`selection_miss_count`)")
    last_selected_at: str | None = Field(default=None, description="Last section-selected timestamp. (`last_selected_at`)")
    last_reviewed_at: str | None = Field(default=None, description="Last section-reviewed timestamp. (`last_reviewed_at`)")
    parked_reason: str | None = Field(default=None, description="Reason for parked/archive status. (`parked_reason`)")
    source_session_id: str | None = Field(default=None, description="출처 세션 ID. (`source_session_id`)")
    created_at: str = Field(description="생성 시각. ISO 8601. (`created_at`)")


class ExtractDecision(BaseModel):
    """fact 추출 결과 + Mem0식 업데이트 의사결정."""

    action: Literal["ADD", "UPDATE", "NOOP", "CONFLICT"] = Field(
        description="ADD=새로 삽입 / UPDATE,CONFLICT=기존 만료 후 새로 삽입 / NOOP=중복으로 스킵. (`action`)"
    )
    content: str = Field(description="추출된 사실 본문(NOOP 포함, 항상 채워짐). (`content`)")
    fact: FactResponse | None = Field(
        default=None,
        description="새로 삽입된 사실(NOOP인 경우 null). (`fact`)",
    )
    expired_id: int | None = Field(
        default=None,
        description="UPDATE/CONFLICT로 valid_until이 채워진 기존 memory_id. (`expired_id`)",
    )
    reason: str | None = Field(default=None, description="판단 사유(LLM). (`reason`)")


class ExtractResult(BaseModel):
    """세션 추출 + 적재 결과 묶음."""

    decisions: list[ExtractDecision] = Field(description="사실별 처리 결과 목록. (`decisions`)")
    future_tasks: list[FutureTaskResponse] = Field(default_factory=list, description="추가 저장된 미래 일정")
    future_tasks_skipped: int = Field(default=0, description="과거/잘못된 시각/원문 검증 실패로 제외된 일정 수")
    counts: dict[str, int] = Field(
        description="action별 카운트(ADD/UPDATE/NOOP/CONFLICT). (`counts`)"
    )
    updated_memory_sections: list[str] = Field(
        default_factory=list,
        description="추출 후 갱신된 user_memory_sections 목록. (`updated_memory_sections`)",
    )
    updated_prompt_caches: list[str] = Field(
        default_factory=list,
        description="호환용 필드. 현재는 updated_memory_sections와 같은 값입니다. (`updated_prompt_caches`)",
    )


class MemoryBlockResponse(BaseModel):
    """Realtime instructions의 `Context` 섹션에 끼워 넣을 메모리 블록 텍스트."""

    text: str = Field(
        description=(
            "OpenAI Realtime 프롬프팅 가이드의 Context 권장 구조 "
            "(Current Insights / Recent Memory / Background / User Profile)로 "
            "조립된 한국어 블록. 그대로 기존 instructions 뒤에 append하면 됨."
        )
    )
    counts: dict[str, int | bool] = Field(
        description="섹션별 항목 수 + 프로필 존재 여부(디버깅용). (`counts`)"
    )
    updated_at: str | None = Field(
        default=None,
        description="Legacy compatibility field. Generated memory blocks are not stored. (`updated_at`)",
    )


class CharacterMemoryPackResponse(BaseModel):
    """active character 기준으로 필터링된 Realtime memory pack."""

    text: str = Field(description="Realtime instructions 뒤에 붙일 memory pack text. (`text`)")
    user: dict[str, Any] = Field(description="memory pack 생성에 사용한 user profile. (`user`)")
    character: dict[str, Any] = Field(description="active character metadata. (`character`)")
    counts: dict[str, int] = Field(description="candidate/included/blocked/section별 개수. (`counts`)")
    included_memory_ids: list[int] = Field(description="pack에 허용된 memory_id 목록. (`included_memory_ids`)")
    blocked_memory_ids: list[int] = Field(description="MAP에서 차단한 memory_id 목록. (`blocked_memory_ids`)")
    cache: dict[str, Any] | None = Field(
        default=None,
        description="캐시 hit 여부와 갱신 정보. (`cache`)",
    )


class ReflectRequest(BaseModel):
    """Reflection 합성 요청."""

    limit: int = Field(
        default=20, ge=2, le=200,
        description="LLM에 넘길 후보 활성 facts 최대 수(최신/중요 순). (`limit`)",
    )
    since: date | None = Field(
        default=None,
        description="이 날짜 이후의 facts만 후보로 사용(없으면 전체). YYYY-MM-DD. (`since`)",
    )
    max_reflections: int = Field(
        default=3, ge=1, le=10,
        description="생성할 reflection 최대 개수. (`max_reflections`)",
    )


class RecallDetailRequest(BaseModel):
    """Realtime function calling에서 쓰는 2단계 상세 기억 검색 요청."""

    query: str = Field(description="검색할 질문 또는 기억 단서. (`query`)")
    active_character_id: str = Field(default="study", description="현재 active character ID. (`active_character_id`)")
    k: int = Field(default=5, ge=1, le=20, description="반환할 memory 최대 개수. (`k`)")
    pool: int = Field(default=30, ge=1, le=200, description="vector 후보 pool 크기. (`pool`)")
    log_decisions: bool = Field(default=False, description="MAP allow/block 결정을 policy_decisions에 기록할지 여부.")


class RecallDetailResponse(BaseModel):
    """MAP 필터를 통과한 상세 기억 검색 결과."""

    query: str = Field(description="입력 query. (`query`)")
    active_character_id: str = Field(description="적용한 active character ID. (`active_character_id`)")
    results: list[dict[str, Any]] = Field(description="MAP 필터를 통과한 검색 결과. (`results`)")
    counts: dict[str, int] = Field(description="retrieved/allowed/returned/blocked 개수. (`counts`)")


class ReflectionResponse(BaseModel):
    """저장된 reflection."""

    reflection_id: int = Field(description="reflection 고유 ID. (`reflection_id`)")
    user_id: str = Field(description="사용자 ID. (`user_id`)")
    character_id: str | None = Field(default=None, description="캐릭터 조건부 통찰이면 해당 character ID. (`character_id`)")
    summary: str = Field(description="합성된 한 문장 통찰. (`summary`)")
    source_memory_ids: list[int] = Field(
        description="이 통찰의 근거가 된 memory_facts.memory_id 목록. (`source_memory_ids`)"
    )
    created_at: str = Field(description="생성 시각. ISO 8601. (`created_at`)")


class ReflectionSearchResult(ReflectionResponse):
    """reflection 벡터 검색 결과."""

    score: float = Field(description="질문과의 코사인 유사도. (`score`)")


class FactSearchResult(FactResponse):
    """검색 결과 사실 + 점수.

    - `ranking=relevance`: `score`는 코사인 유사도. 컴포넌트 필드는 null.
    - `ranking=weighted`: `score`는 relevance·recency·importance 정규화 가중합이고,
      각 컴포넌트(0~1 정규화)도 함께 반환합니다.
    """

    score: float = Field(description="최종 점수(랭킹 모드에 따라 의미가 다름). (`score`)")
    relevance: float | None = Field(default=None, description="정규화된 의미 유사도(weighted 모드). (`relevance`)")
    recency: float | None = Field(default=None, description="정규화된 최신성(weighted 모드). (`recency`)")
    importance_score: float | None = Field(default=None, description="정규화된 중요도(weighted 모드). (`importance_score`)")
    lexical: float | None = Field(default=None, description="날짜/고유명사/키워드 일치 점수(weighted 모드). (`lexical`)")
