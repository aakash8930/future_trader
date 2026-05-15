"""
Database module for RUDRA-ALPHA trading system.
Provides a singleton SQLite database connection with WAL mode and automatic migrations.
"""

import sqlite3
import os
from pathlib import Path
from threading import Lock
from typing import Optional, Any, List, Tuple, Dict
import json
from datetime import datetime

# Database file path
DB_PATH = Path("logs/trading.db")

# Schema version for migrations
SCHEMA_VERSION = 4

# Migration scripts: each version corresponds to the version number
MIGRATIONS = {
    1: """
        -- Initial schema
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            avg_entry REAL NOT NULL,
            qty REAL NOT NULL,
            entry_time TEXT NOT NULL,
            add_count INTEGER DEFAULT 0,
            stop_loss REAL,
            take_profit REAL,
            unrealized_pnl REAL,
            realized_pnl REAL,
            last_updated TEXT NOT NULL,
            post_tp_cooldown INTEGER DEFAULT 0  -- seconds until cooldown expires
        );
        CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
        CREATE INDEX IF NOT EXISTS idx_positions_last_updated ON positions(last_updated);
    """,
    2: """
        -- Ensure index on post_tp_cooldown for quick cooldown checks
        -- (post_tp_cooldown was already in migration 1)
        CREATE INDEX IF NOT EXISTS idx_positions_post_tp_cooldown ON positions(post_tp_cooldown);
    """,
    3: """
        -- Add unique constraint on symbol to prevent duplicate positions for the same symbol
        CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_symbol_unique ON positions(symbol);
    """,
    4: """
        -- Create trades table for complete trade history
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            avg_entry REAL NOT NULL,
            exit_price REAL,
            qty REAL NOT NULL,
            pnl REAL,
            pnl_pct REAL,
            leverage REAL DEFAULT 1.0,
            balance REAL NOT NULL,
            prob REAL NOT NULL,
            threshold REAL NOT NULL,
            atr REAL NOT NULL,
            atr_pct REAL NOT NULL,
            adx REAL NOT NULL,
            regime TEXT NOT NULL,
            stop_loss REAL,
            take_profit REAL,
            exit_reason TEXT,
            close_reason TEXT,
            holding_time_sec INTEGER,
            add_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);
        CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason);
    """
}

class Database:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if self._initialized:
            return
        self._initialized = True
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._run_migrations()

    def _connect(self):
        """Initialize the database connection with WAL mode and busy timeout."""
        try:
            # Ensure logs directory exists
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Connect to SQLite database
            self.conn = sqlite3.connect(
                DB_PATH,
                timeout=10.0,  # Wait up to 10 seconds for lock
                check_same_thread=False  # Allow multiple threads to use the connection
            )
            # Enable WAL mode for better concurrency
            self.conn.execute("PRAGMA journal_mode=WAL;")
            # Set synchronous to NORMAL for better performance while still being safe
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            # Set busy timeout to handle locks
            self.conn.execute("PRAGMA busy_timeout=5000;")  # 5 seconds
            # Enable foreign key constraints
            self.conn.execute("PRAGMA foreign_keys=ON;")
            # Return connections as dictionaries by default
            self.conn.row_factory = sqlite3.Row

            print(f"[DATABASE] Connected to {DB_PATH} in WAL mode.")
        except Exception as e:
            print(f"[DATABASE ERROR] Failed to connect to database: {e}")
            self.conn = None
            raise

    def _run_migrations(self):
        """Run database migrations to bring the schema up to date."""
        if self.conn is None:
            raise RuntimeError("Database connection not initialized")

        cursor = self.conn.cursor()

        # Create schema_version table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
        """)
        self.conn.commit()

        # Get current schema version
        cursor.execute("SELECT MAX(version) FROM schema_version;")
        row = cursor.fetchone()
        current_version = row[0] if row[0] is not None else 0

        print(f"[DATABASE] Current schema version: {current_version}, target: {SCHEMA_VERSION}")

        # Apply migrations in order
        for version in range(current_version + 1, SCHEMA_VERSION + 1):
            if version in MIGRATIONS:
                print(f"[DATABASE] Applying migration to version {version}...")
                try:
                    cursor.executescript(MIGRATIONS[version])
                    cursor.execute("INSERT INTO schema_version (version) VALUES (?);", (version,))
                    self.conn.commit()
                    print(f"[DATABASE] Migration to version {version} applied successfully.")
                except Exception as e:
                    print(f"[DATABASE ERROR] Failed to apply migration {version}: {e}")
                    self.conn.rollback()
                    raise
            else:
                print(f"[DATABASE WARNING] No migration script for version {version}")

    def execute(self, query: str, params: Tuple = ()) -> sqlite3.Cursor:
        """Execute a write query and return the cursor."""
        if self.conn is None:
            raise RuntimeError("Database connection not initialized")
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            return cursor
        except Exception as e:
            print(f"[DATABASE ERROR] Execute failed: {e}")
            print(f"[DATABASE ERROR] Query: {query}")
            print(f"[DATABASE ERROR] Params: {params}")
            self.conn.rollback()
            raise

    def fetchone(self, query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        """Fetch a single row."""
        if self.conn is None:
            raise RuntimeError("Database connection not initialized")
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
        except Exception as e:
            print(f"[DATABASE ERROR] Fetchone failed: {e}")
            raise

    def fetchall(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Fetch all rows."""
        if self.conn is None:
            raise RuntimeError("Database connection not initialized")
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            print(f"[DATABASE ERROR] Fetchall failed: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            print("[DATABASE] Connection closed.")

# Global singleton instance
db = Database()

def get_db() -> Database:
    """Get the database singleton instance."""
    return db