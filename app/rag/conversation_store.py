
import sqlite3
from pathlib import Path


DB_PATH = Path("data/conversations.db")


class ConversationStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)

    def create_conversation(self, conversation_id, title="New conversation"):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conversation_id, title),
            )

    def list_conversations(self):
        with self.connect() as conn:
            rows = conn.execute("""
                SELECT c.id, c.title, c.created_at,
                       (SELECT content FROM messages m
                        WHERE m.conversation_id = c.id
                          AND m.role = 'user'
                        ORDER BY m.id LIMIT 1) AS first_question
                FROM conversations c
                ORDER BY c.created_at DESC, c.rowid DESC
            """).fetchall()
            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            return dict(row) if row else None

    def update_title(self, conversation_id, title):
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )

    def get_messages(self, conversation_id):
        with self.connect() as conn:
            rows = conn.execute("""
                SELECT role, content, sources, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id
            """, (conversation_id,)).fetchall()
            return [dict(row) for row in rows]

    def add_message(self, conversation_id, role, content, sources=None):
        import json

        with self.connect() as conn:
            conn.execute("""
                INSERT INTO messages
                    (conversation_id, role, content, sources)
                VALUES (?, ?, ?, ?)
            """, (
                conversation_id,
                role,
                content,
                json.dumps(sources or []),
            ))
