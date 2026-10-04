from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from fastapi.testclient import TestClient

from server.future_tasks import normalize_task, store_future_tasks, list_future_tasks
from server.memory import extract_session_memories
from server import main as api

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 27, 3, 0, tzinfo=timezone.utc)  # noon in Seoul
QUOTE = "내일 오후 3시 30분에 발표 준비할 거야"
TURNS = [{"role": "user", "text": QUOTE}, {"role": "assistant", "text": "오후 4시에 산책해보세요"}]


def candidate(**overrides):
    return dict(title="발표 준비", due_date="2026-09-28", due_time="15:30",
                time_precision="minute", source_turn_index=0, source_quote=QUOTE) | overrides


class FutureTaskTests(unittest.TestCase):
    def test_exact_minute_and_timezone(self):
        row = normalize_task(candidate(), TURNS, now=NOW)
        self.assertEqual(row["scheduled_at"].isoformat(), "2026-09-28T15:30:00+09:00")
        self.assertFalse(row["needs_clarification"])

    def test_date_only_does_not_invent_midnight(self):
        row = normalize_task(candidate(due_time=None, time_precision="date"), TURNS, now=NOW)
        self.assertEqual(row["due_date"], date(2026, 9, 28))
        self.assertIsNone(row["scheduled_at"])
        self.assertTrue(row["needs_clarification"])

    def test_hour_and_midnight_are_preserved(self):
        for value in ("00:00", "09:00", "15:00"):
            row = normalize_task(candidate(due_time=value, time_precision="hour"), TURNS, now=NOW)
            self.assertEqual(row["due_time"].strftime("%H:%M"), value)
            self.assertIsNotNone(row["scheduled_at"])

    def test_unknown_date_can_keep_known_time(self):
        row = normalize_task(candidate(due_date=None), TURNS, now=NOW)
        self.assertIsNone(row["scheduled_at"])
        self.assertEqual(row["due_time"].minute, 30)
        row = normalize_task(candidate(due_date=None, due_time=None, time_precision="unspecified"), TURNS, now=NOW)
        self.assertTrue(row["needs_clarification"])

    def test_today_future_allowed_past_rejected(self):
        normalize_task(candidate(due_date="2026-09-27", due_time="12:01"), TURNS, now=NOW)
        for changes in ({"due_date": "2026-09-26"},
                        {"due_date": "2026-09-27", "due_time": "12:00"}):
            with self.assertRaises(ValueError):
                normalize_task(candidate(**changes), TURNS, now=NOW)

    def test_timezone_date_boundary(self):
        # Still Sep 27 in UTC, already Sep 28 in Korea.
        with self.assertRaises(ValueError):
            normalize_task(candidate(due_date="2026-09-27", due_time=None, time_precision="date"),
                           TURNS, now=datetime(2026, 9, 27, 16, tzinfo=timezone.utc))

    def test_invalid_dates_times_and_precision(self):
        for changes in ({"due_date": "2026-02-30"}, {"due_time": "25:00"},
                        {"due_time": "15:30:45"}, {"due_time": "15:30+09:00"},
                        {"time_precision": "hour"}, {"due_time": None},
                        {"due_time": None, "time_precision": "unspecified"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_task(candidate(**changes), TURNS, now=NOW)

    def test_assistant_and_invented_evidence_rejected(self):
        for changes in ({"source_turn_index": 1, "source_quote": TURNS[1]["text"]},
                        {"source_turn_index": 7}, {"source_quote": "내일 수술이 있어"}):
            with self.assertRaises(ValueError):
                normalize_task(candidate(**changes), TURNS, now=NOW)

    def test_rephrased_title_has_same_source_identity(self):
        a = normalize_task(candidate(), TURNS, now=NOW)
        b = normalize_task(candidate(title="발표 준비하기"), TURNS, now=NOW)
        self.assertEqual(a["source_key"], b["source_key"])

    def test_extraction_uses_one_call_and_original_session_date(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({"facts": [], "future_tasks": [candidate()]})))])
        with patch("server.memory.get_client", return_value=client):
            facts, tasks = extract_session_memories(TURNS, date(2026, 9, 27), "study")
        self.assertEqual(facts, [])
        self.assertEqual(tasks[0]["due_time"], "15:30")
        client.chat.completions.create.assert_called_once()
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("2026-09-27", prompt)
        self.assertIn('"index": 0', prompt)
        self.assertIn("오전/오후가 불명확", prompt)

    def test_malformed_extraction_is_not_silent_success(self):
        for content in ("not json", '{"facts": []}', '{"facts": {}, "future_tasks": []}'):
            client = Mock()
            client.chat.completions.create.return_value = SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=content))])
            with patch("server.memory.get_client", return_value=client), self.assertRaises(ValueError):
                extract_session_memories(TURNS, date(2026, 9, 27))

    def test_new_read_api_requires_auth_and_valid_date(self):
        client = TestClient(api.app)  # Do not run startup/schema writes.
        response = client.get("/users/example/future-tasks")
        self.assertEqual(response.status_code, 401)
        api.app.dependency_overrides[api.require_api_key] = lambda: None
        try:
            self.assertEqual(client.get("/users/example/future-tasks?task_date=bad").status_code, 422)
        finally:
            api.app.dependency_overrides.clear()

    def test_session_pipeline_stores_tasks_even_without_facts(self):
        conn = Mock()
        conn.execute.return_value.fetchone.return_value = dict(
            session_id="s", character_id="study", session_date=date(2026, 9, 27),
            transcript=TURNS, raw_text=QUOTE)

        @contextmanager
        def get_conn():
            yield conn

        with patch.object(api, "get_conn", get_conn), \
             patch.object(api, "require_user_profile"), \
             patch.object(api, "extract_session_memories", return_value=([], [candidate()])), \
             patch.object(api, "store_facts", return_value=[]), \
             patch.object(api, "store_future_tasks", return_value={"tasks": [], "skipped": 0}) as store, \
             patch.object(api, "refresh_memory_sections_after_fact_update", return_value=[]):
            result = api.extract_session("u", "s")
        store.assert_called_once_with(conn, "u", "s", "study", [candidate()], TURNS)
        self.assertEqual(result["future_tasks_skipped"], 0)


@unittest.skipUnless(os.getenv("FUTURE_TASK_TEST_DATABASE_URL"), "requires explicit isolated PostgreSQL test URL")
class FutureTaskDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ["FUTURE_TASK_TEST_DATABASE_URL"], row_factory=dict_row)
        # All DDL and synthetic rows are rolled back, including this isolated schema.
        schema = "future_test_" + uuid4().hex
        self.conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        self.conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
        self.conn.execute("CREATE TABLE raw_sessions (session_id TEXT PRIMARY KEY)")
        self.conn.execute("INSERT INTO raw_sessions VALUES ('s'), ('other')")
        migration = (ROOT / "server/migrations/001_future_tasks.sql").read_text(encoding="utf-8")
        self.conn.execute(migration)
        self.conn.execute(migration)  # Idempotent migration.

    def tearDown(self):
        self.conn.rollback()
        self.conn.close()

    def save(self, **changes):
        return store_future_tasks(self.conn, "u", "s", "study", [candidate(**changes)], TURNS, now=NOW)

    def test_persist_retry_query_and_tenant_isolation(self):
        row = self.save()["tasks"][0]
        retry = self.save(title="발표 준비하기")["tasks"][0]
        self.assertEqual(row["task_id"], retry["task_id"])
        self.assertEqual(row["scheduled_at"].astimezone(timezone.utc).hour, 6)
        self.assertEqual(len(list_future_tasks(self.conn, "u", ready_only=True)), 1)
        self.assertEqual(list_future_tasks(self.conn, "other"), [])
        self.assertEqual(list_future_tasks(self.conn, "u", character_id="cooking"), [])
        self.assertEqual(list_future_tasks(self.conn, "u", task_date=date(2026, 9, 29)), [])

    def test_date_only_not_ready_and_invalid_not_stored(self):
        self.save(due_time=None, time_precision="date")
        self.assertEqual(len(list_future_tasks(self.conn, "u")), 1)
        self.assertEqual(list_future_tasks(self.conn, "u", ready_only=True), [])
        result = self.save(due_date="2026-09-26")
        self.assertEqual(result["skipped"], 1)

    def test_reextract_does_not_reactivate_completed_task(self):
        self.save()
        self.conn.execute("UPDATE future_tasks SET status='completed'")
        self.save()
        self.assertEqual(list_future_tasks(self.conn, "u"), [])

    def test_one_utterance_can_have_two_occurrences(self):
        turns = [{"role": "user", "text": "내일과 모레 오후 3시 30분에 운동할 거야"}]
        tasks = [candidate(source_quote=turns[0]["text"], due_date=day)
                 for day in ("2026-09-28", "2026-09-29")]
        store_future_tasks(self.conn, "u", "s", "study", tasks, turns, now=NOW)
        self.assertEqual(len(list_future_tasks(self.conn, "u")), 2)

    def test_read_api_serializes_real_database_rows(self):
        self.save()

        @contextmanager
        def get_conn():
            yield self.conn

        api.app.dependency_overrides[api.require_api_key] = lambda: None
        try:
            with patch.object(api, "get_conn", get_conn):
                response = TestClient(api.app).get("/users/u/future-tasks?ready_only=true")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()[0]["due_time"], "15:30:00")
            self.assertNotIn("source_key", response.json()[0])
        finally:
            api.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
