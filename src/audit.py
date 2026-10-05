"""SQLite audit log: every score and every analyst override."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from src.config import AUDIT_DB


SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    p_fraud REAL,
    flag INTEGER,
    threshold REAL,
    is_ood INTEGER NOT NULL DEFAULT 0,
    features TEXT NOT NULL,
    shap_top TEXT,
    shap_vector TEXT,
    review TEXT,
    reviewer_note TEXT,
    case_note TEXT,
    status TEXT NOT NULL DEFAULT 'scored'
);
"""

_JSON_FIELDS = ("features", "shap_top", "shap_vector", "case_note")


def _conn(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path) if path is not None else AUDIT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute(SCHEMA)
    _migrate(con)
    return con


def _migrate(con: sqlite3.Connection) -> None:
    cols = {row[1] for row in con.execute("PRAGMA table_info(scores)")}
    alters = {
        "shap_vector": "ALTER TABLE scores ADD COLUMN shap_vector TEXT",
        "case_note": "ALTER TABLE scores ADD COLUMN case_note TEXT",
        "status": "ALTER TABLE scores ADD COLUMN status TEXT DEFAULT 'scored'",
    }
    for name, sql in alters.items():
        if name not in cols:
            con.execute(sql)
    con.commit()


def _dumps(obj) -> str:
    return json.dumps(obj, allow_nan=False)


def _hydrate(row: sqlite3.Row) -> dict:
    data = dict(row)
    for key in _JSON_FIELDS:
        if data.get(key):
            data[key] = json.loads(data[key])
    return data


def log_score(
    p_fraud: float,
    flag: bool,
    threshold: float,
    is_ood: bool,
    features: dict,
    shap_top: list | None = None,
    shap_vector: list | None = None,
    db_path: Path | None = None,
) -> int:
    con = _conn(db_path)
    cur = con.execute(
        """INSERT INTO scores
           (ts, p_fraud, flag, threshold, is_ood, features, shap_top, shap_vector, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'scored')""",
        (
            time.time(),
            float(p_fraud),
            int(flag),
            float(threshold),
            int(is_ood),
            _dumps(features),
            _dumps(shap_top) if shap_top is not None else None,
            _dumps(shap_vector) if shap_vector is not None else None,
        ),
    )
    con.commit()
    row_id = int(cur.lastrowid)
    con.close()
    return row_id


def log_abstain(features: dict, reason: str, db_path: Path | None = None) -> int:
    """A row the scorer refused. p_fraud stays null so it cannot be read as a score."""
    safe = {k: (None if v is None else v) for k, v in features.items()}
    con = _conn(db_path)
    cur = con.execute(
        """INSERT INTO scores
           (ts, p_fraud, flag, threshold, is_ood, features, shap_top, status, reviewer_note)
           VALUES (?, NULL, NULL, NULL, 0, ?, NULL, 'abstain', ?)""",
        (time.time(), _dumps(safe), reason),
    )
    con.commit()
    row_id = int(cur.lastrowid)
    con.close()
    return row_id


def attach_shap(row_id: int, shap_top: list, shap_vector: list, db_path: Path | None = None) -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE scores SET shap_top = ?, shap_vector = ? WHERE id = ?",
        (_dumps(shap_top), _dumps(shap_vector), int(row_id)),
    )
    con.commit()
    con.close()


def attach_note(row_id: int, note: dict, db_path: Path | None = None) -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE scores SET case_note = ? WHERE id = ?",
        (_dumps(note), int(row_id)),
    )
    con.commit()
    con.close()


def get_score(row_id: int, db_path: Path | None = None) -> dict | None:
    con = _conn(db_path)
    row = con.execute("SELECT * FROM scores WHERE id = ?", (int(row_id),)).fetchone()
    con.close()
    return _hydrate(row) if row is not None else None


def reviewed_vectors(exclude_id: int, db_path: Path | None = None) -> list[dict]:
    """Past analyst decisions that have a SHAP vector. The case agent retrieves over these."""
    con = _conn(db_path)
    rows = con.execute(
        """SELECT id, review, reviewer_note, shap_vector, p_fraud, flag
           FROM scores
           WHERE id != ?
             AND status = 'scored'
             AND review IS NOT NULL
             AND shap_vector IS NOT NULL""",
        (int(exclude_id),),
    ).fetchall()
    con.close()
    return [_hydrate(r) for r in rows]


def review(row_id: int, decision: str, note: str = "", db_path: Path | None = None) -> None:
    if decision not in ("confirm_fraud", "false_alarm", "needs_review"):
        raise ValueError(f"unknown decision {decision}")
    con = _conn(db_path)
    con.execute(
        "UPDATE scores SET review = ?, reviewer_note = ? WHERE id = ?",
        (decision, note, int(row_id)),
    )
    con.commit()
    con.close()


def recent(limit: int = 50, db_path: Path | None = None) -> list[dict]:
    con = _conn(db_path)
    rows = con.execute(
        "SELECT * FROM scores ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]
