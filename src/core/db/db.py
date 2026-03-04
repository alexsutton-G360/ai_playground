from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import psycopg2
import psycopg2.extras

from src.core.config.logging import get_logger

log = get_logger(__name__)

DbName = Literal["proposal", "property", "groupsync", "analytics"]

Params = Union[Tuple[Any, ...], Dict[str, Any], None]


@dataclass(frozen=True)
class PgConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: Optional[str] = None
    application_name: str = "analytics_airflow"

    def dsn(self) -> str:
        # NOTE: DSN may contain password; do not log this.
        parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"dbname={self.dbname}",
            f"user={self.user}",
            f"application_name={self.application_name}",
        ]
        if self.password:  # only include if non-empty
            parts.append(f"password={self.password}")
        return " ".join(parts)

    def safe_display(self) -> str:
        # Safe for logs (never includes password)
        return f"{self.user}@{self.host}:{self.port}/{self.dbname} ({self.application_name})"


def _require_env(name: str) -> str:
    import os

    val = os.getenv(name) or ""
    val = val.strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _optional_env(name: str, default: Optional[str] = "") -> Optional[str]:
    import os

    val = os.getenv(name)
    if val is None or val == "":
        return default.strip() if isinstance(default, str) else default
    return val.strip()


def _db_port() -> int:
    raw = _optional_env("DB_PORT", "5432")
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"DB_PORT must be an integer; got {raw!r}") from e


def _config_for(db: DbName) -> PgConfig:
    default_host = _optional_env("DB_HOST", None)  # <-- key change: don't require
    port = _db_port()

    def _pw(name: str) -> Optional[str]:
        val = _optional_env(name, "")
        return val if val else None

    def _host(prefix: str) -> str:
        specific = _optional_env(f"{prefix}_DB_HOST", None)
        if specific:
            return specific
        if default_host:
            return default_host
        raise RuntimeError(
            f"Missing required env var: {prefix}_DB_HOST (or set DB_HOST as a default)"
        )

    if db == "property":
        return PgConfig(
            host=_host("PROPERTY"),
            port=port,
            dbname=_require_env("PROPERTY_DB_NAME"),
            user=_require_env("PROPERTY_DB_USER"),
            password=_pw("PROPERTY_DB_PASS"),
            application_name="local:property",
        )

    if db == "proposal":
        return PgConfig(
            host=_host("PROPOSAL"),
            port=port,
            dbname=_require_env("PROPOSAL_DB_NAME"),
            user=_require_env("PROPOSAL_DB_USER"),
            password=_pw("PROPOSAL_DB_PASS"),
            application_name="local:proposal",
        )

    if db == "groupsync":
        return PgConfig(
            host=_host("GROUPSYNC"),
            port=port,
            dbname=_require_env("GROUPSYNC_DB_NAME"),
            user=_require_env("GROUPSYNC_DB_USER"),
            password=_pw("GROUPSYNC_DB_PASS"),
            application_name="local:groupsync",
        )

    if db == "analytics":
        return PgConfig(
            host=_host("ANALYTICS"),
            port=port,
            dbname=_require_env("ANALYTICS_DB_NAME"),
            user=_require_env("ANALYTICS_DB_USER"),
            password=_pw("ANALYTICS_DB_PASS"),
            application_name="local:analytics",
        )

    raise ValueError(f"Unknown db alias: {db!r}")


class _ConnCtx:
    """
    Context manager for psycopg2 connections.
    Ensures rollback on exceptions and always closes.
    """

    def __init__(self, db: DbName):
        self.db = db
        self.conn: Optional[psycopg2.extensions.connection] = None

    def __enter__(self):
        cfg = _config_for(self.db)
        log.debug("Connecting [%s] -> %s", self.db, cfg.safe_display())
        self.conn = psycopg2.connect(cfg.dsn())

        # IMPORTANT:
        # Register UUID typecasters on this connection so uuid + uuid[] come back
        # as Python types instead of Postgres array literals like "{...}".
        #
        # This is the root fix for your smoke test "characters instead of GUIDs".
        psycopg2.extras.register_uuid(conn_or_curs=self.conn)

        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if self.conn is not None:
            try:
                if exc is not None:
                    self.conn.rollback()
            finally:
                self.conn.close()


def get_conn(db: DbName) -> _ConnCtx:
    """
    Usage:
        with get_conn("proposal") as conn:
            ...
    """
    return _ConnCtx(db)


# ------------------------------------------------------------
# Read/query helpers (SELECT-style; no commits)
# ------------------------------------------------------------

def query_all(
        db: DbName,
        sql: str,
        params: Params = None,
        *,
        log_timing: bool = True,
) -> list[dict]:
    """
    Run a read-only query (SELECT) and return list[dict] via RealDictCursor.
    """
    start = time.perf_counter()
    with get_conn(db) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = list(cur.fetchall())

    if log_timing:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.debug("Query [%s] returned %d rows in %.1f ms", db, len(rows), elapsed_ms)

    return rows


def query_one(
        db: DbName,
        sql: str,
        params: Params = None,
        *,
        log_timing: bool = True,
) -> Optional[dict]:
    """
    Run a read-only query (SELECT) and return a single row as dict, or None.
    """
    rows = query_all(db, sql, params, log_timing=log_timing)
    return rows[0] if rows else None


# ------------------------------------------------------------
# Write/execute helpers (mutating statements; commit by default)
# ------------------------------------------------------------

def execute(
        db: DbName,
        sql: str,
        params: Params = None,
        *,
        commit: bool = True,
        log_timing: bool = True,
) -> int:
    """
    Run INSERT/UPDATE/DELETE (or DDL). Returns rowcount. Commits by default.
    """
    start = time.perf_counter()
    with get_conn(db) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rowcount = cur.rowcount
        if commit:
            conn.commit()

    if log_timing:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.debug("Execute [%s] rowcount=%d in %.1f ms", db, rowcount, elapsed_ms)

    return rowcount


def execute_returning_one(
        db: DbName,
        sql: str,
        params: Params = None,
        *,
        commit: bool = True,
        log_timing: bool = True,
) -> Optional[dict]:
    """
    Run a mutating statement that returns a single row (e.g. INSERT ... RETURNING).
    Commits by default.

    Returns:
      - dict row if RETURNING produced a row
      - None if no row returned
    """
    start = time.perf_counter()
    with get_conn(db) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        if commit:
            conn.commit()

    if log_timing:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.debug(
            "Execute_returning_one [%s] returned=%s in %.1f ms",
            db,
            bool(row),
            elapsed_ms,
        )

    return dict(row) if row else None


def fetch_df(
        db: DbName,
        sql: str,
        params: Params = None,
):
    """
    Convenience: return a pandas DataFrame if pandas is installed.
    """
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is not installed; use query_all() instead") from e

    rows = query_all(db, sql, params)
    return pd.DataFrame(rows)


def smoke_test() -> Dict[str, Any]:
    """
    Verifies connectivity to each configured DB.
    Returns a dict with db/user/ts or error message.
    """
    out: Dict[str, Any] = {}
    for db in ("proposal", "property", "groupsync", "analytics"):
        try:
            row = query_one(  # type: ignore[arg-type]
                db, "select current_database() as db, current_user as usr, now() as ts;"
            )
            out[db] = row
        except Exception as e:
            out[db] = {"error": str(e)}
    return out
