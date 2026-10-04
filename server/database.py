from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://llm_memory:llm_memory@localhost:5433/llm_memory"
)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _configure(conn: psycopg.Connection) -> None:
    """풀에서 새 커넥션을 만들 때마다 pgvector 타입을 등록한다."""
    register_vector(conn)


pool = ConnectionPool(
    conninfo=DATABASE_URL,
    open=False,
    configure=_configure,
    kwargs={"row_factory": dict_row},
)


def ensure_schema() -> None:
    """schema.sql을 실행해 확장·테이블을 멱등하게 생성한 뒤 풀을 연다.

    pool의 configure(register_vector)는 vector 확장이 존재해야 동작하므로,
    풀을 열기 전에 별도 커넥션으로 스키마부터 적용한다.
    """
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(sql)
        for migration in sorted((SCHEMA_PATH.parent / "migrations").glob("*.sql")):
            conn.execute(migration.read_text(encoding="utf-8"))
    if pool.closed:
        pool.open()


def get_conn():
    """풀에서 커넥션을 빌려주는 컨텍스트 매니저.

    사용: ``with get_conn() as conn: conn.execute(...)``
    블록을 정상 종료하면 트랜잭션이 커밋되고 커넥션은 풀로 반환된다.
    커넥션의 기본 row_factory는 dict_row이다.
    """
    return pool.connection()


def ping_database() -> bool:
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return True
