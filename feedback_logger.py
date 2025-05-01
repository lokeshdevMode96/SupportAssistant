import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            matched_ticket_ids TEXT,
            response TEXT,
            feedback TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_feedback(query, ticket_ids, response, feedback_type):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (query, matched_ticket_ids, response, feedback, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (query, ticket_ids, response, feedback_type, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
