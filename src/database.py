import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="data/stress_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Sessions Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME,
                end_time DATETIME,
                employee_id TEXT
            )
        ''')
        
        # Metrics Table (recorded every N frames or seconds)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp DATETIME,
                hr REAL,
                hrv REAL,
                blinks INTEGER,
                yawns INTEGER,
                dominant_emotion TEXT,
                stress_score REAL,
                stress_level TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')
        
        self.conn.commit()

    def start_session(self, employee_id="Employee_001"):
        cursor = self.conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO sessions (start_time, employee_id)
            VALUES (?, ?)
        ''', (now, employee_id))
        self.conn.commit()
        return cursor.lastrowid

    def end_session(self, session_id):
        cursor = self.conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            UPDATE sessions
            SET end_time = ?
            WHERE session_id = ?
        ''', (now, session_id))
        self.conn.commit()

    def log_metrics(self, session_id, hr, hrv, blinks, yawns, emotion, stress_score, stress_level):
        cursor = self.conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO metrics (session_id, timestamp, hr, hrv, blinks, yawns, dominant_emotion, stress_score, stress_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, now, hr, hrv, blinks, yawns, emotion, stress_score, stress_level))
        self.conn.commit()

    def get_session_data(self, session_id):
        import pandas as pd
        query = f"SELECT * FROM metrics WHERE session_id = {session_id}"
        return pd.read_sql_query(query, self.conn)

    def close(self):
        self.conn.close()
