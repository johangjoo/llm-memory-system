from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterEvalCase:
    case_id: str
    active_character_id: str
    query: str
    expected_external_ids: tuple[str, ...]
    forbidden_owner: str
    expected_switch_to: str | None = None
    memory_eval: bool = True


CASES: tuple[CharacterEvalCase, ...] = (
    CharacterEvalCase(
        case_id="study_eval_metrics",
        active_character_id="study",
        query="발표에서 장기기억 평가 지표와 토큰 비용 이득을 어떻게 설명하지?",
        expected_external_ids=("mem_study_006", "mem_study_007", "mem_shared_010"),
        forbidden_owner="cooking",
    ),
    CharacterEvalCase(
        case_id="study_db_defense",
        active_character_id="study",
        query="MongoDB 대신 PostgreSQL pgvector를 쓰는 근거를 교수님께 어떻게 말하지?",
        expected_external_ids=("mem_shared_013", "mem_study_012", "mem_conflict_001_new"),
        forbidden_owner="cooking",
    ),
    CharacterEvalCase(
        case_id="cooking_fast_recipe",
        active_character_id="cooking",
        query="바쁠 때 15분 안에 만들 수 있는 저녁 메뉴를 추천해줘.",
        expected_external_ids=("mem_cooking_005", "mem_cooking_002", "mem_cooking_004"),
        forbidden_owner="study",
    ),
    CharacterEvalCase(
        case_id="cooking_character_focus",
        active_character_id="cooking",
        query="요리 캐릭터가 답할 때 어떤 식으로 재료와 단계를 나눠야 하지?",
        expected_external_ids=("mem_cooking_020", "mem_cooking_015", "mem_cooking_007"),
        forbidden_owner="study",
    ),
    CharacterEvalCase(
        case_id="switch_study_to_cooking",
        active_character_id="study",
        query="냉장고에 있는 재료로 빠르게 만들 수 있는 요리 알려줘.",
        expected_external_ids=("mem_cooking_004", "mem_cooking_005"),
        forbidden_owner="cooking",
        expected_switch_to="cooking",
        memory_eval=False,
    ),
    CharacterEvalCase(
        case_id="switch_cooking_to_study",
        active_character_id="cooking",
        query="졸업작품 발표에서 장기기억 평가 근거를 어떻게 설명할까?",
        expected_external_ids=("mem_study_006", "mem_study_007"),
        forbidden_owner="study",
        expected_switch_to="study",
        memory_eval=False,
    ),
)


def get_cases(limit: int | None = None) -> tuple[CharacterEvalCase, ...]:
    if limit is None:
        return CASES
    return CASES[: max(0, limit)]
