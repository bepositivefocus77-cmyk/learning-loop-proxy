"""
Database Setup and Migration
--------------------------------
Creates the SQLite schema for a multi-user version of the Learning Loop
Proxy, and migrates existing learning_log.jsonl data into it under a
single "owner" account (you), so no history is lost when moving from the
single-file prototype to the multi-user product.

Run with:
    python setup_database.py
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from passlib.hash import bcrypt

DB_FILE = "learning_loop.db"
OLD_LOG_FILE = "learning_log.jsonl"

OWNER_EMAIL = "owner@local"  # placeholder account that existing data is imported under
OWNER_PASSWORD = "changeme123"  # you should log in and change this after first run


def create_schema(conn):
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


def get_or_create_owner(conn):
    cur = conn.execute("SELECT id FROM users WHERE email = ?", (OWNER_EMAIL,))
    row = cur.fetchone()
    if row:
        return row[0]

    now = datetime.now(timezone.utc).isoformat()
    password_hash = bcrypt.hash(OWNER_PASSWORD)
    cur = conn.execute(
        "INSERT INTO users (email, password_hash, created_at, daily_message_count, daily_count_reset_at) "
        "VALUES (?, ?, ?, 0, ?)",
        (OWNER_EMAIL, password_hash, now, now),
    )
    conn.commit()
    print(f"Created owner account: {OWNER_EMAIL} (password: {OWNER_PASSWORD} -- change this after logging in)")
    return cur.lastrowid


def migrate_old_log(conn, owner_id):
    if not os.path.exists(OLD_LOG_FILE):
        print(f"No existing {OLD_LOG_FILE} found -- nothing to migrate.")
        return 0

    migrated = 0
    skipped = 0

    with open(OLD_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            interaction_id = entry.get("interaction_id") or str(uuid.uuid4())

            existing = conn.execute(
                "SELECT id FROM interactions WHERE interaction_id = ?", (interaction_id,)
            ).fetchone()
            if existing:
                skipped += 1
                continue

            used_context = entry.get("used_context")
            conn.execute(
                """
                INSERT INTO interactions
                    (user_id, interaction_id, timestamp, prompt, response, provider,
                     model_used, feedback, correction, sensitivity, policy_warning, used_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_id,
                    interaction_id,
                    entry.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    entry.get("prompt", ""),
                    entry.get("response", ""),
                    entry.get("provider"),
                    entry.get("model_used"),
                    entry.get("feedback"),
                    entry.get("correction"),
                    entry.get("sensitivity"),
                    entry.get("policy_warning"),
                    json.dumps(used_context) if used_context else None,
                ),
            )
            migrated += 1

    conn.commit()
    return migrated, skipped


def main():
    conn = sqlite3.connect(DB_FILE)
    create_schema(conn)
    owner_id = get_or_create_owner(conn)
    migrated, skipped = migrate_old_log(conn, owner_id)

    print(f"\nDatabase ready at: {os.path.abspath(DB_FILE)}")
    print(f"Migrated {migrated} interactions from {OLD_LOG_FILE} under owner account (user_id={owner_id}).")
    if skipped:
        print(f"Skipped {skipped} already-migrated entries (duplicate interaction_id).")

    conn.close()


if __name__ == "__main__":
    main()