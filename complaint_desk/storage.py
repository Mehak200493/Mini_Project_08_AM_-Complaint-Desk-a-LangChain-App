"""SQLite persistence for complaints (ID generation, search, status updates)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from .config import STATUSES

COLUMNS = [
    "complaint_id",
    "created_at",
    "customer_name",
    "complaint",
    "category",
    "priority",
    "sentiment",
    "response",
    "status",
    "mode",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id  TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    customer_name TEXT DEFAULT '',
    complaint     TEXT NOT NULL,
    category      TEXT NOT NULL,
    priority      TEXT NOT NULL,
    sentiment     TEXT NOT NULL,
    response      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'Open',
    mode          TEXT NOT NULL DEFAULT 'ai'
);
CREATE INDEX IF NOT EXISTS idx_complaints_created ON complaints(created_at);
CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints(category);
"""


class ComplaintStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ create
    def add(
        self,
        *,
        complaint: str,
        category: str,
        priority: str,
        sentiment: str,
        response: str,
        customer_name: str = "",
        mode: str = "ai",
        status: str = "Open",
        created_at: datetime | None = None,
    ) -> str:
        """Insert a complaint and return its generated ID (CMP-YYYY-NNN)."""
        created_at = created_at or datetime.now()
        year = created_at.year
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")  # lock so IDs never collide
            try:
                last = conn.execute(
                    "SELECT COALESCE(MAX(CAST(SUBSTR(complaint_id, 10) AS INTEGER)), 0) "
                    "FROM complaints WHERE complaint_id LIKE ?",
                    (f"CMP-{year}-%",),
                ).fetchone()[0]
                complaint_id = f"CMP-{year}-{last + 1:03d}"
                conn.execute(
                    "INSERT INTO complaints (complaint_id, created_at, customer_name, complaint, "
                    "category, priority, sentiment, response, status, mode) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        complaint_id,
                        created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        customer_name.strip(),
                        complaint,
                        category,
                        priority,
                        sentiment,
                        response,
                        status,
                        mode,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return complaint_id

    # -------------------------------------------------------------------- read
    def get(self, complaint_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM complaints WHERE complaint_id = ? COLLATE NOCASE",
                (complaint_id.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]

    def list_complaints(
        self,
        categories: Sequence[str] | None = None,
        priorities: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
        search: str | None = None,
    ) -> pd.DataFrame:
        """Return complaints (newest first), optionally filtered."""
        clauses: list[str] = []
        params: list[str] = []
        for column, values in (
            ("category", categories),
            ("priority", priorities),
            ("status", statuses),
        ):
            if values:
                clauses.append(f"{column} IN ({','.join('?' * len(values))})")
                params.extend(values)
        if search and search.strip():
            like = f"%{search.strip()}%"
            clauses.append("(complaint_id LIKE ? OR complaint LIKE ? OR customer_name LIKE ?)")
            params.extend([like, like, like])

        sql = "SELECT * FROM complaints"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, complaint_id DESC"

        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if df.empty:
            return pd.DataFrame(columns=COLUMNS).astype({"created_at": "datetime64[ns]"})
        df["created_at"] = pd.to_datetime(df["created_at"])
        return df

    # ------------------------------------------------------------------ update
    def update_status(self, complaint_id: str, status: str) -> bool:
        if status not in STATUSES:
            raise ValueError(f"Invalid status: {status!r}")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE complaints SET status = ? WHERE complaint_id = ?",
                (status, complaint_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------ delete
    def clear(self) -> None:
        """Remove all complaints (used by the demo-data script)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM complaints")
