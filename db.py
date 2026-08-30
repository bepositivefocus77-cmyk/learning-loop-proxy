"""
Database access layer for the multi-user Learning Loop Proxy.
All reads/writes to SQLite go through this module.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning_loop.db")
DAILY_MESSAGE_LIMIT = 30


def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            daily_message_count INTEGER NOT NULL DEFAULT 0,
            daily_count_reset_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            interaction_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            provider TEXT,
            model_used TEXT,
            feedback TEXT,
            correction TEXT,
            sensitivity TEXT,
            policy_warning TEXT,
            used_context TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions (user_id)")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

def create_user(email: str, password_hash: str):
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    try:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, created_at, daily_message_count, daily_count_reset_at) "
            "VALUES (?, ?, ?, 0, ?)",
            (email, password_hash, now, now),
        )
        conn.commit()
        user_id = cur.lastrowid
        return user_id
    finally:
        conn.close()


def get_user_by_email(email: str):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def check_and_consume_daily_quota(user_id: int) -> tuple[bool, int, str | None]:
    """
    Checks whether the user still has quota left today, resetting the
    counter if 24+ hours have passed since the last reset. If quota is
    available, increments the counter and returns (True, remaining, None).
    If not, returns (False, 0, reset_time_iso) without incrementing.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT daily_message_count, daily_count_reset_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError("User not found")

        count = row["daily_message_count"]
        reset_at = datetime.fromisoformat(row["daily_count_reset_at"])
        now = datetime.now(timezone.utc)

        if now - reset_at >= timedelta(hours=24):
            count = 0
            reset_at = now

        if count >= DAILY_MESSAGE_LIMIT:
            conn.execute(
                "UPDATE users SET daily_message_count = ?, daily_count_reset_at = ? WHERE id = ?",
                (count, reset_at.isoformat(), user_id),
            )
            conn.commit()
            next_reset = reset_at + timedelta(hours=24)
            return False, 0, next_reset.isoformat()

        count += 1
        conn.execute(
            "UPDATE users SET daily_message_count = ?, daily_count_reset_at = ? WHERE id = ?",
            (count, reset_at.isoformat(), user_id),
        )
        conn.commit()
        remaining = DAILY_MESSAGE_LIMIT - count
        return True, remaining, None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Interaction management
# ---------------------------------------------------------------------------

def insert_interaction(user_id, prompt, response, provider, model_used,
                        sensitivity=None, policy_warning=None, used_context=None):
    interaction_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO interactions
                (user_id, interaction_id, timestamp, prompt, response, provider,
                 model_used, feedback, correction, sensitivity, policy_warning, used_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
            """,
            (
                user_id,
                interaction_id,
                datetime.now(timezone.utc).isoformat(),
                prompt,
                response,
                provider,
                model_used,
                sensitivity,
                policy_warning,
                json.dumps(used_context) if used_context else None,
            ),
        )
        conn.commit()
        return interaction_id
    finally:
        conn.close()


def get_interaction(interaction_id: str, user_id: int):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM interactions WHERE interaction_id = ? AND user_id = ?",
            (interaction_id, user_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_feedback(interaction_id: str, user_id: int, feedback: str, correction: str | None):
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE interactions SET feedback = ?, correction = ? WHERE interaction_id = ? AND user_id = ?",
            (feedback, correction, interaction_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_all_interactions_for_user(user_id: int):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM interactions WHERE user_id = ? ORDER BY timestamp",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()