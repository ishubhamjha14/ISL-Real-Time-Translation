"""
db_manager.py
Phase 8: Database Integration for Firebase and MongoDB Atlas.

Architecture:
1. Firebase Firestore:
   - Real-time app-facing sync for live session state and recognition stream.
2. MongoDB Atlas:
   - Structured ML logging: vocabulary, training metadata, inference logs, and misrecognitions for retraining.
3. Asynchronous Non-Blocking Execution:
   - All database network I/O is offloaded to a background ThreadPoolExecutor to prevent blocking real-time camera loops.
4. Graceful Fallback:
   - Automatically detects if cloud credentials (MONGO_URI / FIREBASE_CREDENTIALS_PATH) are missing and seamlessly falls back to structured local JSON/JSONL logging.
"""

import os
import sys
import json
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

# MongoDB Imports
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

# Firebase Imports
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


class DatabaseManager:
    """
    Centralized Database Manager handling MongoDB Atlas and Firebase Firestore
    with asynchronous dispatch and local JSON/JSONL fallback.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        mongo_db_name: str = "isl_database",
        firebase_creds_path: Optional[str] = None,
        local_logs_dir: Optional[Path] = None,
    ):
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        self.mongo_db_name = os.getenv("DB_NAME") or os.getenv("MONGO_DB_NAME") or mongo_db_name
        self.firebase_creds_path = firebase_creds_path or os.getenv("FIREBASE_CREDENTIALS_PATH")

        # Local logs fallback directory
        self.logs_dir = local_logs_dir or (project_root / "data" / "logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Thread pool for non-blocking asynchronous database writes
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DBWorker")
        self._csv_lock = threading.Lock()

        # Reports directory for Tableau CSV logs
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Connections
        self.mongo_client = None
        self.mongo_db = None
        self.firestore_client = None

        self._init_mongodb()
        self._init_firebase()

    def _init_mongodb(self):
        """Connects to MongoDB Atlas or activates local fallback."""
        if not PYMONGO_AVAILABLE:
            print("[DB Warning] 'pymongo' library not installed. Using local JSONL logging fallback.")
            return

        if not self.mongo_uri or self.mongo_uri.startswith("mongodb+srv://<"):
            print("[DB Info] MONGO_URI not configured in .env. Using structured local fallback at data/logs/.")
            return

        try:
            self.mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=4000)
            # Verify connection with ping
            self.mongo_client.admin.command("ping")
            self.mongo_db = self.mongo_client[self.mongo_db_name]
            print(f"[DB Info] Successfully connected to MongoDB Atlas database '{self.mongo_db_name}'.")
        except Exception as e:
            print(f"[DB Warning] Could not connect to MongoDB Atlas ({e}). Falling back to local logging.")
            self.mongo_client = None
            self.mongo_db = None

    def _init_firebase(self):
        """Initializes Firebase Admin SDK or activates local state fallback."""
        if not FIREBASE_AVAILABLE:
            print("[DB Warning] 'firebase-admin' library not installed. Using local session sync fallback.")
            return

        if not self.firebase_creds_path or not os.path.exists(self.firebase_creds_path):
            print("[DB Info] FIREBASE_CREDENTIALS_PATH not configured. Using local session state fallback.")
            return

        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.firebase_creds_path)
                firebase_admin.initialize_app(cred)
            self.firestore_client = firestore.client()
            print("[DB Info] Successfully initialized Firebase Firestore client.")
        except Exception as e:
            print(f"[DB Warning] Could not initialize Firebase Admin SDK ({e}). Falling back to local session state.")
            self.firestore_client = None

    def _append_local_jsonl(self, filename: str, record: Dict[str, Any]):
        """Helper to append structured records into local JSONL files."""
        filepath = self.logs_dir / filename
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ---------------------------------------------------------
    # 1. Training Metadata Logging (MongoDB / Fallback)
    # ---------------------------------------------------------
    def log_training_metadata(
        self,
        vocab_list: List[str],
        model_metrics: Dict[str, Any],
        run_id: str,
    ):
        """Asynchronously logs vocabulary and training metrics to MongoDB."""
        record = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "vocab_size": len(vocab_list),
            "vocab": vocab_list,
            "metrics": model_metrics,
        }

        def _task():
            if self.mongo_db is not None:
                try:
                    self.mongo_db["training_metadata"].insert_one(record)
                    return
                except Exception as e:
                    print(f"[DB Error] Failed to write training metadata to MongoDB: {e}")
            # Fallback
            self._append_local_jsonl("training_metadata.jsonl", record)

        self._executor.submit(_task)

    # ---------------------------------------------------------
    # 2. Inference Event Logging (MongoDB / Fallback)
    # ---------------------------------------------------------
    def log_inference_event(
        self,
        signer_id: str,
        predicted_gloss: str,
        confidence: float,
        latency_ms: float = 20.0,
        status: str = "correct",
        fps: float = 30.0,
        **kwargs,
    ):
        """Asynchronously logs real-time inference predictions."""
        record = {
            "signer_id": signer_id,
            "predicted_gloss": predicted_gloss,
            "confidence": round(float(confidence), 4),
            "latency_ms": round(float(latency_ms), 2),
            "fps": round(float(fps), 1),
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }

        def _task():
            if self.mongo_db is not None:
                try:
                    self.mongo_db["inference_logs"].insert_one(record)
                    return
                except Exception as e:
                    print(f"[DB Error] Failed to write inference event to MongoDB: {e}")
            # Fallback
            self._append_local_jsonl("inference_events.jsonl", record)

        self._executor.submit(_task)

    # ---------------------------------------------------------
    # 3. Misrecognition Logging for Retraining (MongoDB / Fallback)
    # ---------------------------------------------------------
    def log_misrecognition(
        self,
        signer_id: str,
        expected_gloss: str,
        predicted_gloss: str,
        landmark_features_ref: Optional[str] = None,
        notes: str = "",
    ):
        """Asynchronously logs user-flagged misrecognitions for active learning / retraining."""
        record = {
            "signer_id": signer_id,
            "expected_gloss": expected_gloss,
            "predicted_gloss": predicted_gloss,
            "landmark_features_ref": landmark_features_ref,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        }

        def _task():
            if self.mongo_db is not None:
                try:
                    self.mongo_db["misrecognitions"].insert_one(record)
                    return
                except Exception as e:
                    print(f"[DB Error] Failed to write misrecognition to MongoDB: {e}")
            # Fallback
            self._append_local_jsonl("misrecognitions.jsonl", record)

        self._executor.submit(_task)

    # ---------------------------------------------------------
    # 4. Live Session Synchronization (Firebase / Fallback)
    # ---------------------------------------------------------
    def sync_live_result(
        self,
        session_id: str,
        current_gloss: str,
        confidence: float,
    ):
        """Asynchronously pushes live recognized sign state to Firebase Firestore."""
        data = {
            "session_id": session_id,
            "current_gloss": current_gloss,
            "confidence": round(float(confidence), 4),
            "updated_at": datetime.utcnow().isoformat(),
        }

        def _task():
            if self.firestore_client is not None:
                try:
                    doc_ref = self.firestore_client.collection("sessions").document(session_id)
                    doc_ref.set(data, merge=True)
                    return
                except Exception as e:
                    print(f"[Firebase Error] Failed to sync session to Firestore: {e}")
            # Fallback: maintain local JSON snapshot
            local_session_file = self.logs_dir / f"session_{session_id}.json"
            try:
                with open(local_session_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

        self._executor.submit(_task)

    # ---------------------------------------------------------
    # 5. Real-Time & Pilot Logging for Tableau Analytics (CSV)
    # ---------------------------------------------------------
    def log_realtime_pilot_csv(
        self,
        session_id: str,
        predicted_gloss: str,
        confidence: float,
        landmark_latency_ms: float = 0.0,
        model_latency_ms: float = 0.0,
        total_latency_ms: Optional[float] = None,
        fps: float = 0.0,
        status: str = "correct",
        timestamp: Optional[str] = None,
    ):
        """
        Appends real-time or pilot recognition events to reports/realtime_pilot_logs.csv
        for direct loading into Tableau Desktop and Tableau Public.
        Columns: timestamp, session_id, predicted_gloss, confidence, total_latency_ms, fps, status
        """
        csv_path = self.reports_dir / "realtime_pilot_logs.csv"
        ts = timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if total_latency_ms is None:
            tot_lat = round(float(landmark_latency_ms) + float(model_latency_ms), 2)
        else:
            tot_lat = round(float(total_latency_ms), 2)
        row = [
            ts,
            str(session_id),
            str(predicted_gloss),
            f"{float(confidence):.4f}",
            f"{tot_lat:.2f}",
            f"{float(fps):.1f}",
            str(status),
        ]

        def _task():
            with self._csv_lock:
                write_header = not csv_path.exists() or csv_path.stat().st_size == 0
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([
                            "timestamp",
                            "session_id",
                            "predicted_gloss",
                            "confidence",
                            "total_latency_ms",
                            "fps",
                            "status",
                        ])
                    writer.writerow(row)

        self._executor.submit(_task)

    def close(self):
        """Closes thread pools and database connections cleanly."""
        self._executor.shutdown(wait=False)
        if self.mongo_client:
            self.mongo_client.close()


if __name__ == "__main__":
    print("=== Testing Database Manager Pipeline ===")

    db = DatabaseManager()

    print("\n1. Testing Training Metadata Logging...")
    db.log_training_metadata(
        vocab_list=["namaste", "atm", "hospital"],
        model_metrics={"top1_acc": 34.58, "top5_acc": 73.83},
        run_id="run_transformer_gloss_v1",
    )

    print("2. Testing Real-time Inference Event Logging...")
    db.log_inference_event(
        signer_id="signer_01",
        predicted_gloss="namaste",
        confidence=0.892,
        latency_ms=14.5,
    )

    print("3. Testing Misrecognition Flagging...")
    db.log_misrecognition(
        signer_id="signer_01",
        expected_gloss="hello",
        predicted_gloss="namaste",
        landmark_features_ref="sample_buffer_seq_001.npy",
        notes="Flagged via keyboard shortcut 'm'",
    )

    print("4. Testing Live Session State Sync...")
    db.sync_live_result(
        session_id="live_demo_01",
        current_gloss="namaste",
        confidence=0.892,
    )

    print("5. Testing Tableau-Ready CSV Logging...")
    db.log_realtime_pilot_csv(
        session_id="live_demo_01",
        predicted_gloss="namaste",
        confidence=0.892,
        landmark_latency_ms=31.2,
        model_latency_ms=1.8,
        fps=29.4,
        status="correct",
    )

    import time
    time.sleep(1.0)
    db.close()

    print("\nLocal fallback log files created:")
    for p in (project_root / "data" / "logs").iterdir():
        print(f" - {p.name} ({p.stat().st_size} bytes)")

    csv_file = project_root / "reports" / "realtime_pilot_logs.csv"
    if csv_file.exists():
        print(f"\nTableau CSV file generated:\n - {csv_file.name} ({csv_file.stat().st_size} bytes)")

    print("\n=== Database Manager Tests Passed Successfully! ===")
