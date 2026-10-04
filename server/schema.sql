-- MindMate 장기기억 스키마 (PostgreSQL + pgvector)
-- 멱등: 서버 startup(ensure_schema)과 docker init 양쪽에서 안전하게 재실행 가능.
-- 미래 일정 테이블은 migrations/001_future_tasks.sql에서 추가한다.
-- ensure_schema와 docker init은 이 파일 다음에 해당 마이그레이션을 실행한다.

CREATE EXTENSION IF NOT EXISTS vector;

-- 사용자 프로필 (user_id는 TEXT; eval harness가 question_id를 user_id로 쓰므로 FK 강제는 피함)
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    user_name   TEXT NOT NULL,
    age         INT,
    birth_date  DATE,
    persona     TEXT,
    robot_name  TEXT,
    job         TEXT,
    living_info TEXT,
    location    TEXT,
    habit       TEXT,
    api_key     TEXT NOT NULL,
    -- Realtime prompt-ready memory caches live in user_memory_sections.
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_name ON users(user_name);

-- Remove legacy users.memory_prompt columns if an older database has them.
ALTER TABLE users DROP COLUMN IF EXISTS memory_prompt;
ALTER TABLE users DROP COLUMN IF EXISTS memory_prompt_updated_at;

-- L0: 원본 세션
CREATE TABLE IF NOT EXISTS raw_sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    character_id TEXT,
    session_date DATE,
    transcript   JSONB,
    raw_text     TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_user ON raw_sessions(user_id);
ALTER TABLE raw_sessions ADD COLUMN IF NOT EXISTS character_id TEXT;

-- 캐릭터 정의. MVP는 study/cooking 두 캐릭터를 사용한다.
CREATE TABLE IF NOT EXISTS characters (
    character_id    TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    system_prompt   TEXT,
    switch_rules    TEXT,
    allowed_domains TEXT[],
    blocked_domains TEXT[],
    tone            TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE characters ADD COLUMN IF NOT EXISTS system_prompt TEXT;
ALTER TABLE characters ADD COLUMN IF NOT EXISTS switch_rules TEXT;

-- L1: 추출된 사실 (망각축 valid_until 포함)
CREATE TABLE IF NOT EXISTS memory_facts (
    memory_id     BIGSERIAL PRIMARY KEY,
    external_id   TEXT,
    user_id       TEXT NOT NULL,
    content       TEXT NOT NULL,
    summary_for_prompt TEXT,
    memory_type   TEXT,
    scope         TEXT NOT NULL DEFAULT 'shared',
    owner_character_id TEXT,
    domain_tags   TEXT[],
    importance    INT,
    event_time    DATE,
    valid_until   DATE,
    validity_status TEXT NOT NULL DEFAULT 'current',
    confidence    REAL DEFAULT 1.0,
    sensitivity   TEXT NOT NULL DEFAULT 'normal',
    conflict_group_id TEXT,
    selection_status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (selection_status IN ('candidate', 'selected', 'parked', 'archived', 'blocked')),
    selection_score REAL,
    selected_count INT NOT NULL DEFAULT 0,
    selection_miss_count INT NOT NULL DEFAULT 0,
    last_selected_at TIMESTAMPTZ,
    last_reviewed_at TIMESTAMPTZ,
    parked_reason TEXT,
    source_session_id TEXT REFERENCES raw_sessions(session_id),
    embedding     VECTOR(1536),
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS external_id TEXT;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS summary_for_prompt TEXT;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS scope TEXT NOT NULL DEFAULT 'shared';
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS owner_character_id TEXT;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS validity_status TEXT NOT NULL DEFAULT 'current';
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 1.0;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS sensitivity TEXT NOT NULL DEFAULT 'normal';
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS conflict_group_id TEXT;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS selection_status TEXT NOT NULL DEFAULT 'candidate';
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS selection_score REAL;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS selected_count INT NOT NULL DEFAULT 0;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS selection_miss_count INT NOT NULL DEFAULT 0;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS last_selected_at TIMESTAMPTZ;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS last_reviewed_at TIMESTAMPTZ;
ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS parked_reason TEXT;
UPDATE memory_facts
SET selection_status = CASE
        WHEN scope = 'blocked' OR sensitivity = 'sensitive' THEN 'blocked'
        ELSE 'archived'
    END,
    last_reviewed_at = COALESCE(last_reviewed_at, NOW()),
    parked_reason = COALESCE(parked_reason, 'schema backfill: excluded from prompt selection')
WHERE (scope = 'blocked'
       OR sensitivity = 'sensitive'
       OR validity_status = 'outdated'
       OR valid_until <= CURRENT_DATE)
  AND selection_status NOT IN ('archived', 'blocked');
CREATE INDEX IF NOT EXISTS idx_facts_user ON memory_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_scope_owner ON memory_facts(user_id, scope, owner_character_id);
CREATE INDEX IF NOT EXISTS idx_facts_conflict ON memory_facts(user_id, conflict_group_id);
CREATE INDEX IF NOT EXISTS idx_facts_selection
    ON memory_facts(user_id, selection_status, scope, owner_character_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_external_id
    ON memory_facts(user_id, external_id)
    WHERE external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_facts_vec  ON memory_facts USING hnsw (embedding vector_cosine_ops);

-- Remove legacy whole-character prompt cache if an older database has it.
DROP TABLE IF EXISTS user_character_prompts;

-- User-specific prompt-ready memory sections.
-- memory_facts remains the source of truth. These rows are the compact sections
-- used at Realtime start: shared + one active character section.
CREATE TABLE IF NOT EXISTS user_memory_sections (
    user_id              TEXT NOT NULL,
    section_key          TEXT NOT NULL CHECK (section_key IN ('shared', 'study', 'cooking')),
    section_text         TEXT NOT NULL DEFAULT '',
    included_memory_ids  BIGINT[] NOT NULL DEFAULT '{}',
    blocked_memory_ids   BIGINT[] NOT NULL DEFAULT '{}',
    counts               JSONB NOT NULL DEFAULT '{}'::jsonb,
    memory_limit         INT NOT NULL DEFAULT 0,
    max_section_chars    INT NOT NULL DEFAULT 2500,
    build_mode           TEXT NOT NULL DEFAULT 'score_pruned_sections',
    updated_from_session_id TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, section_key)
);
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS section_text TEXT NOT NULL DEFAULT '';
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS included_memory_ids BIGINT[] NOT NULL DEFAULT '{}';
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS blocked_memory_ids BIGINT[] NOT NULL DEFAULT '{}';
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS counts JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS memory_limit INT NOT NULL DEFAULT 0;
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS max_section_chars INT NOT NULL DEFAULT 2500;
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS build_mode TEXT NOT NULL DEFAULT 'score_pruned_sections';
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS updated_from_session_id TEXT;
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE user_memory_sections ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
CREATE INDEX IF NOT EXISTS idx_user_memory_sections_updated
    ON user_memory_sections(user_id, updated_at DESC);

-- L3: reflection. character_id가 null이면 shared insight, 값이 있으면 캐릭터 조건부 insight.
CREATE TABLE IF NOT EXISTS reflections (
    reflection_id     BIGSERIAL PRIMARY KEY,
    user_id           TEXT NOT NULL,
    character_id      TEXT,
    summary           TEXT,
    source_memory_ids BIGINT[],
    embedding         VECTOR(1536),
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE reflections ADD COLUMN IF NOT EXISTS character_id TEXT;
CREATE INDEX IF NOT EXISTS idx_refl_user ON reflections(user_id);
CREATE INDEX IF NOT EXISTS idx_refl_character ON reflections(user_id, character_id);

-- MAP(Memory Access Policy) 결정 로그. 평가와 발표 방어용 근거로 사용한다.
CREATE TABLE IF NOT EXISTS policy_decisions (
    decision_id         BIGSERIAL PRIMARY KEY,
    user_id             TEXT NOT NULL,
    active_character_id TEXT,
    query               TEXT,
    memory_id           BIGINT,
    decision            TEXT NOT NULL,
    reason              TEXT,
    score               REAL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_policy_decisions_user
    ON policy_decisions(user_id, active_character_id, created_at DESC);

-- 실험 결과 로그 (검증 하네스용)
CREATE TABLE IF NOT EXISTS eval_results (
    id                BIGSERIAL PRIMARY KEY,
    test_name         TEXT,
    dataset           TEXT,
    question_id       TEXT,
    question_type     TEXT,
    user_id           TEXT,
    model_answer      TEXT,
    gold_answer       TEXT,
    correct           INT,
    prompt_tokens     INT,
    completion_tokens INT,
    latency_sec       REAL,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
