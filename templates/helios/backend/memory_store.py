"""
Helios Memory Store - SQLite-based persistent memory system
Handles model metadata, training journals, and future context storage.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    SQLite-based memory store for persistent data management.
    Handles model metadata, training sessions, and future context storage.
    """

    def __init__(self, db_path: str = "helios_memory.db"):
        """
        Initialize the memory store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self.conn = None  # For persistent in-memory connection

        if self.db_path == ":memory:":
            # For in-memory databases, we need a single, persistent connection
            # to keep the database alive for the duration of the object's life.
            self.conn = sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        else:
            # Ensure database directory exists for file-based databases
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._initialize_schema()

        logger.info(f"MemoryStore initialized with database: {db_path}")

    def create_all_tables(self):
        """Public method to create all database tables - alias for _initialize_schema."""
        self._initialize_schema()
        logger.info("Database schema initialized successfully")

    def close(self):
        """Close the persistent database connection if it exists."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("In-memory database connection closed.")

    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Enhanced journal entries with metadata
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS enhanced_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    confidence_score REAL,
                    success_metric REAL,
                    context_hash TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    archived BOOLEAN DEFAULT FALSE
                )
            """
            )

            # Knowledge fragments for persistent learning
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_fragments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    fragment_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Performance metrics tracking
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Memory compaction logs
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    details TEXT,
                    items_affected INTEGER,
                    space_saved INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # ...existing tables...
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            """
            )

            # Training sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    progress INTEGER DEFAULT 0,
                    config TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Training logs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    metrics TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES training_sessions (job_id)
                )
            """
            )

            # Model predictions table (for future analysis)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    prediction_data TEXT NOT NULL,
                    confidence REAL,
                    actual_outcome TEXT,
                    prediction_timestamp TEXT NOT NULL,
                    draw_date TEXT,
                    is_correct BOOLEAN
                )
            """
            )

            # Context storage table (for future phases)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS context_storage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_type TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    context_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    UNIQUE(context_type, context_key)
                )
            """
            )

            # System events table (for monitoring and debugging)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp TEXT NOT NULL,
                    level TEXT DEFAULT 'INFO'
                )
            """
            )

            conn.commit()
            logger.info("Database schema initialized successfully")

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        # Use the persistent connection if it exists (for in-memory dbs)
        if self.conn:
            yield self.conn
            return

        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    # Model Management Methods

    def save_model_metadata(
        self,
        name: str,
        file_path: str,
        architecture: str,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save model metadata to the database.

        Returns:
            Model ID
        """
        with self.lock:
            timestamp = datetime.now().isoformat()
            metadata_json = json.dumps(metadata or {})

            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO models
                    (name, file_path, architecture, version, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        name,
                        file_path,
                        architecture,
                        version,
                        timestamp,
                        timestamp,
                        metadata_json,
                    ),
                )

                model_id = cursor.lastrowid
                conn.commit()

                if model_id is None:
                    raise RuntimeError("Failed to get model ID after insertion")

                logger.info(f"Saved model metadata: {name} (ID: {model_id})")
                return model_id

    def get_model_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM models WHERE name = ? AND is_active = 1
            """,
                (name,),
            )

            row = cursor.fetchone()
            if row:
                result = dict(row)
                result["metadata"] = json.loads(result["metadata"] or "{}")
                return result

            return None

    def list_models(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all models in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM models"
            if active_only:
                query += " WHERE is_active = 1"
            query += " ORDER BY created_at DESC"

            cursor.execute(query)

            models = []
            for row in cursor.fetchall():
                model = dict(row)
                model["metadata"] = json.loads(model["metadata"] or "{}")
                models.append(model)

            return models

    def delete_model(self, name: str) -> bool:
        """Soft delete a model (mark as inactive)."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE models SET is_active = 0, updated_at = ?
                    WHERE name = ?
                """,
                    (datetime.now().isoformat(), name),
                )

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    logger.info(f"Model {name} marked as inactive")

                return success

    # Training Session Management

    def create_training_session(
        self, job_id: str, model_name: str, config: Dict[str, Any]
    ) -> int:
        """Create a new training session record."""
        with self.lock:
            timestamp = datetime.now().isoformat()
            config_json = json.dumps(config)

            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO training_sessions
                    (job_id, model_name, status, start_time, config, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        job_id,
                        model_name,
                        "started",
                        timestamp,
                        config_json,
                        timestamp,
                        timestamp,
                    ),
                )

                session_id = cursor.lastrowid
                conn.commit()

                if session_id is None:
                    raise RuntimeError("Failed to get session ID after insertion")

                logger.info(f"Created training session: {job_id} (ID: {session_id})")
                return session_id

    def update_training_session(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
    ):
        """Update training session status and progress."""
        with self.lock:
            timestamp = datetime.now().isoformat()

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Build dynamic update query
                updates = ["updated_at = ?"]
                params: List[Any] = [timestamp]

                if status is not None:
                    updates.append("status = ?")
                    params.append(status)

                    if status in ["completed", "failed"]:
                        updates.append("end_time = ?")
                        params.append(timestamp)

                if progress is not None:
                    updates.append("progress = ?")
                    params.append(progress)

                if error_message is not None:
                    updates.append("error_message = ?")
                    params.append(error_message)

                params.append(job_id)

                query = f"UPDATE training_sessions SET {', '.join(updates)} WHERE job_id = ?"
                cursor.execute(query, params)

                conn.commit()

    def get_training_session(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training session by job ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM training_sessions WHERE job_id = ?
            """,
                (job_id,),
            )

            row = cursor.fetchone()
            if row:
                result = dict(row)
                result["config"] = json.loads(result["config"] or "{}")
                return result

            return None

    def add_training_log(
        self,
        job_id: str,
        epoch: int,
        loss: float,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Add a training log entry."""
        with self.lock:
            timestamp = datetime.now().isoformat()
            metrics_json = json.dumps(metrics or {})

            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO training_logs
                    (job_id, epoch, loss, metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (job_id, epoch, loss, metrics_json, timestamp),
                )

                conn.commit()

    def get_training_logs(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all training logs for a job."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM training_logs
                WHERE job_id = ?
                ORDER BY epoch ASC
            """,
                (job_id,),
            )

            logs = []
            for row in cursor.fetchall():
                log = dict(row)
                log["metrics"] = json.loads(log["metrics"] or "{}")
                logs.append(log)

            return logs

    # Prediction Management

    def save_prediction(
        self,
        model_name: str,
        prediction_data: Dict[str, Any],
        confidence: float,
        draw_date: Optional[str] = None,
    ) -> int:
        """Save a model prediction."""
        with self.lock:
            timestamp = datetime.now().isoformat()
            prediction_json = json.dumps(prediction_data)

            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO predictions
                    (model_name, prediction_data, confidence, prediction_timestamp, draw_date)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (model_name, prediction_json, confidence, timestamp, draw_date),
                )

                prediction_id = cursor.lastrowid
                conn.commit()

                if prediction_id is None:
                    raise RuntimeError("Failed to get prediction ID after insertion")

                return prediction_id

    def update_prediction_outcome(
        self, prediction_id: int, actual_outcome: Dict[str, Any], is_correct: bool
    ):
        """Update prediction with actual outcome."""
        with self.lock:
            outcome_json = json.dumps(actual_outcome)

            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE predictions
                    SET actual_outcome = ?, is_correct = ?
                    WHERE id = ?
                """,
                    (outcome_json, is_correct, prediction_id),
                )

                conn.commit()

    def get_model_predictions(
        self, model_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent predictions for a model."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM predictions
                WHERE model_name = ?
                ORDER BY prediction_timestamp DESC
                LIMIT ?
            """,
                (model_name, limit),
            )

            predictions = []
            for row in cursor.fetchall():
                prediction = dict(row)
                prediction["prediction_data"] = json.loads(
                    prediction["prediction_data"]
                )
                if prediction["actual_outcome"]:
                    prediction["actual_outcome"] = json.loads(
                        prediction["actual_outcome"]
                    )
                predictions.append(prediction)

            return predictions

    # Context Storage (for future phases)

    def store_context(
        self,
        context_type: str,
        context_key: str,
        context_data: Dict[str, Any],
        expires_at: Optional[datetime] = None,
    ):
        """Store context data for future use."""
        with self.lock:
            timestamp = datetime.now().isoformat()
            expires_iso = expires_at.isoformat() if expires_at else None
            data_json = json.dumps(context_data)

            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO context_storage
                    (context_type, context_key, context_data, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (context_type, context_key, data_json, timestamp, expires_iso),
                )

                conn.commit()

    def get_context(
        self, context_type: str, context_key: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve context data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM context_storage
                WHERE context_type = ? AND context_key = ?
                AND (expires_at IS NULL OR expires_at > ?)
            """,
                (context_type, context_key, datetime.now().isoformat()),
            )

            row = cursor.fetchone()
            if row:
                result = dict(row)
                result["context_data"] = json.loads(result["context_data"])
                return result

            return None

    def cleanup_expired_context(self):
        """Remove expired context data."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM context_storage
                    WHERE expires_at IS NOT NULL
                    AND expires_at <= ?
                """,
                    (datetime.now().isoformat(),),
                )

                deleted = cursor.rowcount
                conn.commit()

                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} expired context entries")

    # System Events and Monitoring

    def log_event(
        self,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ):
        """Log a system event."""
        with self.lock:
            timestamp = datetime.now().isoformat()
            data_json = json.dumps(event_data or {})

            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO system_events
                    (event_type, event_data, timestamp, level)
                    VALUES (?, ?, ?, ?)
                """,
                    (event_type, data_json, timestamp, level),
                )

                conn.commit()

    def get_recent_events(
        self, event_type: Optional[str] = None, hours: int = 24, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent system events."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if event_type:
                cursor.execute(
                    """
                    SELECT * FROM system_events
                    WHERE event_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (event_type, since, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM system_events
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (since, limit),
                )

            events = []
            for row in cursor.fetchall():
                event = dict(row)
                event["event_data"] = json.loads(event["event_data"])
                events.append(event)

            return events

    # ============================================
    # PHASE 3: Enhanced Memory Management Methods
    # ============================================

    def store_enhanced_journal_entry(
        self,
        model_name: str,
        session_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        confidence_score: Optional[float] = None,
        success_metric: Optional[float] = None,
        context_hash: Optional[str] = None,
    ) -> int:
        """Store an enhanced journal entry with metacognitive information."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO enhanced_journal
                (model_name, session_id, event_type, event_data, confidence_score, success_metric, context_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model_name,
                    session_id,
                    event_type,
                    json.dumps(event_data),
                    confidence_score,
                    success_metric,
                    context_hash,
                ),
            )

            entry_id = cursor.lastrowid
            conn.commit()

            if entry_id is None:
                raise RuntimeError("Failed to get entry ID after insertion")

            logger.info(
                f"Stored enhanced journal entry {entry_id} for model {model_name}"
            )
            return entry_id

    def get_enhanced_journal_entries(
        self,
        model_name: Optional[str] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve enhanced journal entries with filtering options."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM enhanced_journal WHERE 1=1"
            params = []

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if not include_archived:
                query += " AND archived = FALSE"

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            entries = []

            for row in cursor.fetchall():
                entry = {
                    "id": row[0],
                    "model_name": row[1],
                    "session_id": row[2],
                    "event_type": row[3],
                    "event_data": json.loads(row[4]),
                    "confidence_score": row[5],
                    "success_metric": row[6],
                    "context_hash": row[7],
                    "timestamp": row[8],
                    "archived": bool(row[9]),
                }
                entries.append(entry)

            return entries

    def store_knowledge_fragment(
        self,
        model_name: str,
        fragment_type: str,
        content: str,
        relevance_score: float = 1.0,
    ) -> int:
        """Store a knowledge fragment for future retrieval."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO knowledge_fragments
                (model_name, fragment_type, content, relevance_score)
                VALUES (?, ?, ?, ?)
            """,
                (model_name, fragment_type, content, relevance_score),
            )

            fragment_id = cursor.lastrowid
            conn.commit()

            if fragment_id is None:
                raise RuntimeError("Failed to get fragment ID after insertion")

            logger.info(
                f"Stored knowledge fragment {fragment_id} for model {model_name}"
            )
            return fragment_id

    def get_knowledge_fragments(
        self,
        model_name: Optional[str] = None,
        fragment_type: Optional[str] = None,
        min_relevance: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve knowledge fragments with filtering options."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM knowledge_fragments WHERE relevance_score >= ?"
            params: List[Any] = [min_relevance]

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            if fragment_type:
                query += " AND fragment_type = ?"
                params.append(fragment_type)

            query += " ORDER BY relevance_score DESC, last_accessed DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            fragments = []

            for row in cursor.fetchall():
                fragment = {
                    "id": row[0],
                    "model_name": row[1],
                    "fragment_type": row[2],
                    "content": row[3],
                    "relevance_score": row[4],
                    "usage_count": row[5],
                    "last_accessed": row[6],
                    "created_at": row[7],
                }
                fragments.append(fragment)

            return fragments

    def update_knowledge_fragment_usage(self, fragment_id: int):
        """Update usage statistics for a knowledge fragment."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE knowledge_fragments
                SET usage_count = usage_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (fragment_id,),
            )

            conn.commit()

    def store_performance_metric(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        context: Optional[str] = None,
    ) -> int:
        """Store a performance metric for analysis."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO performance_metrics
                (model_name, metric_name, metric_value, context)
                VALUES (?, ?, ?, ?)
            """,
                (model_name, metric_name, metric_value, context),
            )

            metric_id = cursor.lastrowid
            conn.commit()

            if metric_id is None:
                raise RuntimeError("Failed to get metric ID after insertion")

            logger.info(
                f"Stored performance metric {metric_name} for model {model_name}: {metric_value}"
            )
            return metric_id

    def get_performance_metrics(
        self,
        model_name: Optional[str] = None,
        metric_name: Optional[str] = None,
        hours: int = 168,  # Default: last week
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Retrieve performance metrics with filtering options."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM performance_metrics WHERE timestamp >= ?"
            params: List[Any] = [since]

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            metrics = []

            for row in cursor.fetchall():
                metric = {
                    "id": row[0],
                    "model_name": row[1],
                    "metric_name": row[2],
                    "metric_value": row[3],
                    "context": row[4],
                    "timestamp": row[5],
                }
                metrics.append(metric)

            return metrics

    def log_memory_operation(
        self,
        operation_type: str,
        details: str,
        items_affected: int = 0,
        space_saved: int = 0,
    ) -> int:
        """Log a memory management operation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO memory_operations
                (operation_type, details, items_affected, space_saved)
                VALUES (?, ?, ?, ?)
            """,
                (operation_type, details, items_affected, space_saved),
            )

            operation_id = cursor.lastrowid
            conn.commit()

            if operation_id is None:
                raise RuntimeError("Failed to get operation ID after insertion")

            logger.info(f"Logged memory operation: {operation_type} - {details}")
            return operation_id

    def compact_memory(
        self, archive_days: int = 30, relevance_threshold: float = 0.1
    ) -> Dict[str, int]:
        """Perform memory compaction by archiving old and low-relevance data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=archive_days)).isoformat()
            stats = {
                "archived_journal_entries": 0,
                "deleted_knowledge_fragments": 0,
                "cleaned_performance_metrics": 0,
            }

            # Archive old journal entries
            cursor.execute(
                """
                UPDATE enhanced_journal
                SET archived = TRUE
                WHERE timestamp < ? AND archived = FALSE
            """,
                (cutoff_date,),
            )
            stats["archived_journal_entries"] = cursor.rowcount

            # Delete low-relevance knowledge fragments
            cursor.execute(
                """
                DELETE FROM knowledge_fragments
                WHERE relevance_score < ? AND usage_count = 0
            """,
                (relevance_threshold,),
            )
            stats["deleted_knowledge_fragments"] = cursor.rowcount

            # Clean old performance metrics (keep aggregated summaries)
            old_cutoff = (datetime.now() - timedelta(days=90)).isoformat()
            cursor.execute(
                """
                DELETE FROM performance_metrics
                WHERE timestamp < ?
            """,
                (old_cutoff,),
            )
            stats["cleaned_performance_metrics"] = cursor.rowcount

            conn.commit()

            # Log the compaction operation
            self.log_memory_operation(
                operation_type="compaction",
                details=f"Archived {stats['archived_journal_entries']} journal entries, "
                f"deleted {stats['deleted_knowledge_fragments']} knowledge fragments, "
                f"cleaned {stats['cleaned_performance_metrics']} performance metrics",
                items_affected=sum(stats.values()),
            )

            logger.info(f"Memory compaction completed: {stats}")
            return stats

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Count records in each table
            tables = [
                "enhanced_journal",
                "knowledge_fragments",
                "performance_metrics",
                "memory_operations",
            ]
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Get memory usage info
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]

            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]

            stats["database_size_bytes"] = page_size * page_count
            stats["database_size_mb"] = round(
                stats["database_size_bytes"] / (1024 * 1024), 2
            )

            # Get recent activity
            cursor.execute(
                """
                SELECT COUNT(*) FROM enhanced_journal
                WHERE timestamp > datetime('now', '-24 hours')
            """
            )
            stats["journal_entries_24h"] = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM performance_metrics
                WHERE timestamp > datetime('now', '-24 hours')
            """
            )
            stats["metrics_24h"] = cursor.fetchone()[0]

            return stats

    def vacuum_database(self):
        """Optimize database storage."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                logger.info("Database vacuumed successfully")
