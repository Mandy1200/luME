import sqlite3
import json
import time
from typing import Dict, Any

TELEMETRY_DB = "lume_telemetry.db"

def init_telemetry_db():
    conn = sqlite3.connect(TELEMETRY_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            session_id TEXT,
            node_name TEXT,
            status TEXT,
            message TEXT,
            metrics TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_telemetry_event(session_id: str, node_name: str, status: str, message: str, metrics: Dict[str, Any] = None):
    """
    Logs an event in the sqlite telemetry database.
    """
    init_telemetry_db()
    conn = sqlite3.connect(TELEMETRY_DB)
    cursor = conn.cursor()
    
    metrics_str = json.dumps(metrics) if metrics else "{}"
    cursor.execute("""
        INSERT INTO telemetry_logs (timestamp, session_id, node_name, status, message, metrics)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (time.time(), session_id, node_name, status, message, metrics_str))
    
    conn.commit()
    conn.close()
    print(f"📊 Telemetry logged: {node_name} -> {status} ({message})")

def fetch_telemetry_logs() -> list:
    conn = sqlite3.connect(TELEMETRY_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM telemetry_logs ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs
