-- Additive migration: independent from memory_facts and its cache lifecycle.
CREATE TABLE IF NOT EXISTS future_tasks (
    task_id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_session_id TEXT NOT NULL REFERENCES raw_sessions(session_id),
    source_turn_index INT NOT NULL CHECK (source_turn_index >= 0),
    source_quote TEXT NOT NULL,
    source_key TEXT NOT NULL,
    character_id TEXT,
    title TEXT NOT NULL,
    due_date DATE,
    due_time TIME,
    timezone TEXT NOT NULL,
    time_precision TEXT NOT NULL CHECK (time_precision IN ('unspecified', 'date', 'hour', 'minute')),
    scheduled_at TIMESTAMPTZ,
    needs_clarification BOOLEAN NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'cancelled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, source_session_id, source_key),
    CHECK ((scheduled_at IS NULL) = needs_clarification),
    CHECK ((due_time IS NULL AND time_precision IN ('date', 'unspecified'))
        OR (due_time IS NOT NULL AND time_precision IN ('hour', 'minute'))),
    CHECK (scheduled_at IS NULL OR (due_date IS NOT NULL AND due_time IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_future_tasks_due
    ON future_tasks (user_id, scheduled_at) WHERE status = 'pending' AND scheduled_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_future_tasks_date ON future_tasks (user_id, due_date);
