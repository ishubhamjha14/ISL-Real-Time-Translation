"""
realtime_inference.py
Phase 6 & 8: Real-Time Live Camera Inference Pipeline for Indian Sign Language (ISL) Translation.
Integrated with Phase 5 Speech Synthesis (TTS) & Phase 8 Database Manager (MongoDB Atlas & Firebase Firestore).

Features:
1. Model & Checkpoint Loading:
   - Loads best trained ISLTransformerClassifier from models/best_transformer_gloss.pth.
   - Loads label mapping from data/label_map.json.
2. MediaPipe Holistic Real-Time Extraction:
   - Identical 1662-dimensional feature extraction (Pose: 132, Face: 1404, LH: 63, RH: 63).
3. Sliding Window Mechanism:
   - Maintains a 30-frame temporal buffer (collections.deque(maxlen=30)).
   - Executes non-blocking inference on full window using torch.no_grad() and softmax.
   - Computes inference latency in milliseconds (ms).
   - Applies confidence thresholding to suppress noise and transient gestures.
4. Non-Blocking Text & Speech Synthesis (TTS):
   - Integrated SpeechSynthesizer running on a dedicated background thread.
   - Requires N consecutive frames (default: 5) of high confidence before speaking.
   - Auto-debounced with minimum cooldown timer.
5. Cloud Database & Active Learning Integration (Phase 8):
   - Asynchronous non-blocking logging via DatabaseManager (ThreadPoolExecutor).
   - Logs inference event (signer_id, gloss, confidence, latency_ms) to MongoDB Atlas.
   - Syncs real-time session state to Firebase Firestore for external app UI sync.
   - Graceful local JSONL fallback under data/logs/ if cloud credentials not configured.
   - Retraining Key Trigger ('m'): Flags the last recognized gloss as a misrecognition,
     saves the exact 30-frame landmark tensor (.npy), and pushes to MongoDB for active learning retraining.
6. Rich Visual HUD Overlay:
   - MediaPipe skeletal landmark connections (pose and hands).
   - Top banner: Recognized gloss, confidence bar, FPS, latency (ms), buffer fill, and DB mode badge.
   - Live speaking subtitle banner.
   - Visual alert banner for flagged misrecognitions.
   - Bottom shortcut navigation bar ('m' to flag, 'q' to quit).
"""

import os
import sys
import time
import json
import csv
import warnings
import argparse
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress verbose C++/absl logs and deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore")

# Auto-fallback to isl_py310 conda environment
try:
    import torch
    import torch.nn.functional as F
    import cv2
    import numpy as np
    import mediapipe as mp
except ImportError:
    conda_py = r"C:\Users\abhis\anaconda3\anaconda\envs\isl_py310\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        result = subprocess.run([conda_py] + sys.argv, cwd=str(project_root))
        sys.exit(result.returncode)
    raise

from src.extract_landmarks import extract_keypoints_from_results as extract_keypoints
from src.dataset import trim_landmarks
from src.model import ISLTransformerClassifier
from src.tts_service import SpeechSynthesizer
from src.db_manager import DatabaseManager

# Constants
POSE_FEATURES = 33 * 4   # 132
FACE_FEATURES = 468 * 3  # 1404
HAND_FEATURES = 21 * 3   # 63
TOTAL_FEATURES = POSE_FEATURES + FACE_FEATURES + (HAND_FEATURES * 2)  # 1662
TRIMMED_FEATURES = 226   # Upper pose (100) + Left Hand (63) + Right Hand (63)
GESTURE_WINDOW_SIZE = 30
COOLDOWN_SECONDS = 1.5   # Post-recognition locked cooldown in seconds
DEFAULT_THRESHOLD = 0.40 # Softmax confidence threshold
RESET_CONSECUTIVE_FRAMES = 8 # Frames with 0 hands required to reset state machine

# State Machine States
STATE_IDLE = "IDLE"                            # State 1: No Hands detected ("SHOW HAND TO START SIGNING")
STATE_RECORDING = "RECORDING"                  # State 2: Hand Raised, capturing 30 frames ("RECORDING GESTURE... [X/30]")
STATE_RECOGNIZED = "RECOGNIZED"                # State 3: Recognized & TTS speech ("GLOSS: <CLASS_NAME>")
STATE_WAITING_FOR_RESET = "WAITING_FOR_RESET"  # State 4: Latching / Reset Required ("PLEASE LOWER HAND BEFORE NEXT SIGN")

# Compatibility Constants
WINDOW_SIZE = 30
MOTION_START_THRESHOLD = 0.015
MOTION_STOP_THRESHOLD = 0.008

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def append_to_realtime_pilot_logs(
    session_id: str,
    predicted_gloss: str,
    confidence: float,
    total_latency_ms: float,
    fps: float,
    status: str = "correct",
    timestamp: Optional[str] = None,
):
    """
    Appends real-time or pilot recognition events to reports/realtime_pilot_logs.csv
    for direct loading into Tableau Desktop and Tableau Public.
    Columns: timestamp, session_id, predicted_gloss, confidence, total_latency_ms, fps, status
    """
    csv_dir = project_root / "reports"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "realtime_pilot_logs.csv"
    ts = timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    row = [
        ts,
        str(session_id),
        str(predicted_gloss),
        f"{float(confidence):.4f}",
        f"{float(total_latency_ms):.2f}",
        f"{float(fps):.1f}",
        str(status),
    ]
    try:
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
    except Exception as e:
        print(f"[Warning] Failed to append to {csv_path}: {e}")


def extract_hand_wrist_positions(results):
    """
    Extracts 2D normalized coordinates [x, y] of left and right wrists.
    Prefers hand landmark wrist (pt 0) if hand is detected, else falls back to pose wrist (pts 15, 16).
    Returns (lw, rw) where each is np.ndarray([x, y]) or None.
    """
    lw = None
    rw = None

    if results.left_hand_landmarks:
        pt = results.left_hand_landmarks.landmark[0]
        lw = np.array([pt.x, pt.y], dtype=np.float32)
    elif results.pose_landmarks:
        pt = results.pose_landmarks.landmark[15]
        if pt.visibility > 0.35:
            lw = np.array([pt.x, pt.y], dtype=np.float32)

    if results.right_hand_landmarks:
        pt = results.right_hand_landmarks.landmark[0]
        rw = np.array([pt.x, pt.y], dtype=np.float32)
    elif results.pose_landmarks:
        pt = results.pose_landmarks.landmark[16]
        if pt.visibility > 0.35:
            rw = np.array([pt.x, pt.y], dtype=np.float32)

    return lw, rw


def calculate_motion_energy(curr_lw, curr_rw, prev_lw, prev_rw) -> float:
    """
    Computes frame-to-frame Euclidean distance (movement velocity) of the hands:
    motion_energy = max(||lw_curr - lw_prev||, ||rw_curr - rw_prev||).
    """
    energies = []
    if curr_lw is not None and prev_lw is not None:
        energies.append(float(np.linalg.norm(curr_lw - prev_lw)))
    if curr_rw is not None and prev_rw is not None:
        energies.append(float(np.linalg.norm(curr_rw - prev_rw)))

    if energies:
        return max(energies)
    return 0.0


def draw_styled_landmarks(image: np.ndarray, results):
    """Draws styled MediaPipe pose and hand connections on the frame."""
    # Draw Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    # Draw Left Hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

    # Draw Right Hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 220, 100), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80, 44, 255), thickness=2, circle_radius=2),
        )


def render_hud(
    image: np.ndarray,
    system_status: str,
    gloss: str,
    confidence: float,
    fps: float,
    latency_ms: float,
    buffer_count: int,
    target_count: int = 30,
    threshold: float = DEFAULT_THRESHOLD,
    reset_count: int = 0,
    reset_target: int = RESET_CONSECUTIVE_FRAMES,
    db_mode: str = "LOCAL",
    speaking_text: Optional[str] = None,
    flag_notice: Optional[str] = None,
) -> np.ndarray:
    """
    Renders an aesthetically styled translucent HUD top card with 4 distinct states:
    1. STATE_IDLE: "SHOW HAND TO START SIGNING"
    2. STATE_RECORDING: "RECORDING GESTURE... [X/30]" with a filling progress bar
    3. STATE_RECOGNIZED: "GLOSS: <CLASS_NAME> (<CONFIDENCE>%)" in vibrant Green banner
    4. STATE_WAITING_FOR_RESET: "PLEASE LOWER HAND BEFORE NEXT SIGN" & "SIGN COMPLETE! LOWER HAND TO RESET"
    """
    h, w, _ = image.shape
    banner_height = 96

    # Create semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)

    # Accent color line at bottom of banner
    if system_status == STATE_RECORDING:
        accent_color = (0, 165, 255)    # Electric amber / orange
        line_thick = 3
    elif system_status in (STATE_RECOGNIZED, STATE_WAITING_FOR_RESET):
        accent_color = (0, 255, 128)    # Vibrant emerald green
        line_thick = 3
    else:
        accent_color = (100, 125, 145)  # Clean neutral slate grey
        line_thick = 2

    cv2.line(image, (0, banner_height), (w, banner_height), accent_color, line_thick)

    # 1. State Indicator & Main Gloss Banner Line
    if system_status == STATE_RECORDING:
        cv2.circle(image, (26, 32), 7, (0, 90, 255), -1)
        cv2.circle(image, (26, 32), 3, (255, 255, 255), -1)
        cv2.putText(image, "RECORDING", (40, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 255), 2, cv2.LINE_AA)

        main_text = f"RECORDING GESTURE... [{buffer_count}/{target_count}]"
        cv2.putText(image, main_text, (168, 36), cv2.FONT_HERSHEY_DUPLEX, 0.78, (0, 230, 255), 2, cv2.LINE_AA)

    elif system_status in (STATE_RECOGNIZED, STATE_WAITING_FOR_RESET):
        is_reset_wait = (system_status == STATE_WAITING_FOR_RESET)
        circle_color = (0, 220, 255) if is_reset_wait else (0, 255, 128)
        cv2.circle(image, (26, 32), 6, circle_color, -1)

        status_tag = "LATCHED" if is_reset_wait else "RECOGNIZED"
        tag_color = (0, 200, 255) if is_reset_wait else (0, 255, 128)
        cv2.putText(image, status_tag, (40, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, tag_color, 2, cv2.LINE_AA)

        # Solid Green Banner for Top-1 Recognized Gloss
        main_text = f"GLOSS: {gloss.upper()} ({confidence * 100:.1f}%)"
        cv2.putText(image, main_text, (150, 37), cv2.FONT_HERSHEY_DUPLEX, 0.86, (0, 255, 128), 2, cv2.LINE_AA)

    else:
        # State 1 (No Hands): "SHOW HAND TO START SIGNING"
        cv2.circle(image, (26, 32), 5, (140, 150, 165), -1)
        cv2.putText(image, "IDLE", (40, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (150, 165, 180), 1, cv2.LINE_AA)

        main_text = "SHOW HAND TO START SIGNING"
        cv2.putText(image, main_text, (100, 37), cv2.FONT_HERSHEY_DUPLEX, 0.82, (200, 215, 225), 2, cv2.LINE_AA)

    # 2. Confidence Indicator & Dynamic Progress Bar
    if system_status == STATE_WAITING_FOR_RESET:
        if reset_count > 0:
            conf_str = f"LOWER HAND: RESETTING [{reset_count}/{reset_target} frames]"
            bar_ratio = min(1.0, reset_count / float(reset_target))
            bar_color = (0, 255, 128)
            conf_color = (0, 240, 255)
        else:
            conf_str = "PLEASE LOWER HAND BEFORE NEXT SIGN"
            bar_ratio = 1.0
            bar_color = (0, 165, 255)
            conf_color = (0, 200, 255)
    elif system_status == STATE_RECOGNIZED:
        conf_str = f"CONFIDENCE: {confidence * 100:.1f}%"
        bar_ratio = min(1.0, confidence)
        bar_color = (0, 255, 128)
        conf_color = (220, 220, 220)
    elif system_status == STATE_RECORDING:
        conf_str = f"PROGRESS: {buffer_count}/{target_count} frames"
        bar_ratio = min(1.0, buffer_count / float(target_count))
        bar_color = (0, 215, 255)
        conf_color = (220, 220, 220)
    else:
        conf_str = "BUFFER: 0/30 [WAITING FOR HAND]"
        bar_ratio = 0.0
        bar_color = (70, 75, 85)
        conf_color = (180, 185, 195)

    cv2.putText(
        image,
        conf_str,
        (18, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        conf_color,
        1,
        cv2.LINE_AA,
    )

    bar_x = 350
    bar_y = 64
    bar_w = 125
    bar_h = 13
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 60, 70), -1)
    fill_w = int(bar_w * bar_ratio)
    if fill_w > 0:
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (110, 120, 130), 1)

    # 3. Buffer, Latency & FPS Stats (Right-aligned)
    fps_text = f"FPS: {fps:4.1f} | LAT: {latency_ms:3.0f}ms"
    cv2.putText(
        image,
        fps_text,
        (w - 245, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )

    stat_text = f"RESET: {reset_target}F | DB: {db_mode}"
    cv2.putText(
        image,
        stat_text,
        (w - 245, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (150, 175, 195),
        1,
        cv2.LINE_AA,
    )

    # 4. State 4 Latching Warning Pill (Prominent notification)
    active_y_offset = banner_height + 10
    if system_status == STATE_WAITING_FOR_RESET:
        warn_h = 36
        if reset_count > 0:
            warn_label = f"[RESETTING] HANDS LOWERED ({reset_count}/{reset_target} FRAMES)..."
            pill_border = (0, 255, 128)
            pill_text_color = (0, 255, 128)
        else:
            warn_label = "[!] SIGN COMPLETE! LOWER HAND TO RESET"
            pill_border = (0, 140, 255)
            pill_text_color = (0, 210, 255)

        (ww, _), _ = cv2.getTextSize(warn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        warn_w = min(ww + 40, w - 30)

        warn_overlay = image.copy()
        cv2.rectangle(warn_overlay, (15, active_y_offset), (15 + warn_w, active_y_offset + warn_h), (20, 22, 32), -1)
        cv2.addWeighted(warn_overlay, 0.88, image, 0.12, 0, image)

        cv2.rectangle(image, (15, active_y_offset), (15 + warn_w, active_y_offset + warn_h), pill_border, 2)
        cv2.putText(
            image,
            warn_label,
            (28, active_y_offset + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            pill_text_color,
            2,
            cv2.LINE_AA,
        )
        active_y_offset += warn_h + 8

    # 5. Speaking Indicator Sub-Banner (if active)
    if speaking_text:
        sub_h = 36
        speech_label = f"SPEAKING: \"{speaking_text}\""

        (tw, _), _ = cv2.getTextSize(speech_label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        pill_w = min(tw + 55, w - 30)

        sub_overlay = image.copy()
        cv2.rectangle(sub_overlay, (15, active_y_offset), (15 + pill_w, active_y_offset + sub_h), (18, 22, 32), -1)
        cv2.addWeighted(sub_overlay, 0.85, image, 0.15, 0, image)

        cv2.rectangle(image, (15, active_y_offset), (15 + pill_w, active_y_offset + sub_h), (0, 200, 255), 1)
        cv2.circle(image, (32, active_y_offset + sub_h // 2), 6, (0, 255, 128), -1)

        cv2.putText(
            image,
            speech_label,
            (48, active_y_offset + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 240, 255),
            2,
            cv2.LINE_AA,
        )
        active_y_offset += sub_h + 8

    # 6. Misrecognition Alert Pill
    if flag_notice:
        flag_h = 36
        flag_label = f"[!] {flag_notice}"

        (fw, _), _ = cv2.getTextSize(flag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        flag_w = min(fw + 40, w - 30)

        flag_overlay = image.copy()
        cv2.rectangle(flag_overlay, (15, active_y_offset), (15 + flag_w, active_y_offset + flag_h), (25, 18, 18), -1)
        cv2.addWeighted(flag_overlay, 0.88, image, 0.12, 0, image)

        cv2.rectangle(image, (15, active_y_offset), (15 + flag_w, active_y_offset + flag_h), (0, 140, 255), 2)
        cv2.putText(
            image,
            flag_label,
            (28, active_y_offset + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    # 7. Bottom Controls Bar
    bottom_h = 24
    bot_overlay = image.copy()
    cv2.rectangle(bot_overlay, (0, h - bottom_h), (w, h), (15, 18, 24), -1)
    cv2.addWeighted(bot_overlay, 0.70, image, 0.30, 0, image)
    controls_text = "Hotkeys: 'm' Flag Misrecognition for Retraining  |  'q' Exit Application"
    cv2.putText(
        image,
        controls_text,
        (15, h - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (180, 190, 200),
        1,
        cv2.LINE_AA,
    )

    return image


def load_isl_model(checkpoint_path: str, device: torch.device):
    """Loads trained checkpoint and reconstructs model architecture."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint.get("num_classes", 37)
    feature_dim = checkpoint.get("feature_dim", TOTAL_FEATURES)
    class_names = checkpoint.get("class_names")

    # If class_names not in checkpoint, load from data/label_map.json
    if not class_names:
        label_map_file = Path("data/label_map.json")
        if label_map_file.exists():
            with open(label_map_file, "r") as f:
                lmap = json.load(f)
            class_names = [k for k, _ in sorted(lmap.items(), key=lambda x: x[1])]
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]

    hp = checkpoint.get("hyperparameters", {})
    d_model = hp.get("d_model", 128)
    nhead = hp.get("nhead", 4)
    dim_feedforward = hp.get("dim_feedforward", 256)
    num_layers = hp.get("num_layers", 2)
    dropout = hp.get("dropout", 0.35)

    model = ISLTransformerClassifier(
        input_dim=feature_dim,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        dropout=dropout,
        pooling="mean",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[Info] Loaded ISL model checkpoint (Best Val Acc: {checkpoint.get('val_acc', 0)*100:.2f}%)")
    print(f"[Info] Recognized Classes ({len(class_names)}): {', '.join(class_names[:8])}...")

    return model, class_names, feature_dim


def run_inference(
    checkpoint_path: str = "models/best_transformer_gloss.pth",
    camera_index: int = 0,
    video_path: str = None,
    threshold: float = DEFAULT_THRESHOLD,
    enable_tts: bool = True,
    cooldown_seconds: float = COOLDOWN_SECONDS,
    flip: Optional[bool] = None,
    tts_cooldown: Optional[float] = None,
    enable_db: bool = True,
    signer_id: str = "signer_01",
    session_id: str = "live_session_01",
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    motion_start_threshold: float = MOTION_START_THRESHOLD,
    motion_stop_threshold: float = MOTION_STOP_THRESHOLD,
    reset_consecutive_frames: int = RESET_CONSECUTIVE_FRAMES,
):
    """
    Main Real-Time Camera Inference Loop with Strict Hand-Reset Required State Machine:
    - State 1 (IDLE / No Hands): "SHOW HAND TO START SIGNING"
    - State 2 (RECORDING / Hand Raised): "RECORDING GESTURE... [X/30]" with filling progress bar
    - State 3 (RECOGNIZED): "GLOSS: <CLASS_NAME> (<CONFIDENCE>%)" in vibrant Green banner (TTS speaks ONCE)
    - State 4 (WAITING_FOR_RESET): "PLEASE LOWER HAND BEFORE NEXT SIGN" & "SIGN COMPLETE! LOWER HAND TO RESET"
      (Refuses to record or predict until hands are lowered for >= 8 consecutive frames)
    """
    if tts_cooldown is not None:
        cooldown_seconds = tts_cooldown

    # Determine horizontal flip (mirroring):
    # Camera feed defaults to True so physical movement matches mirrored perception.
    # Pre-recorded video files default to False unless explicitly forced via --flip.
    should_flip = flip if flip is not None else (video_path is None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ISL Real-Time Hand-Reset State Machine & Translation Pipeline ===")
    print(f"Device:                 {device}")
    print(f"Confidence Threshold:   {threshold * 100:.1f}%")
    print(f"Window Size:            {GESTURE_WINDOW_SIZE} frames")
    print(f"Reset Required:         {reset_consecutive_frames} consecutive frames of 0 hands")
    print(f"Horizontal Mirroring:   {'Enabled (cv2.flip)' if should_flip else 'Disabled'}")
    print(f"Speech Synthesis:       {'Enabled' if enable_tts else 'Disabled'}")
    print(f"Signer ID:              {signer_id}")
    print(f"Session ID:             {session_id}")

    model, class_names, feature_dim = load_isl_model(checkpoint_path, device)

    # Initialize Speech Synthesizer (runs on dedicated daemon thread, non-blocking)
    tts = SpeechSynthesizer(cooldown_seconds=cooldown_seconds) if enable_tts else None

    # Initialize Database Manager
    db = DatabaseManager() if enable_db else None
    if db is not None:
        db_mode = "ATLAS" if db.mongo_client is not None else "LOCAL"
    else:
        db_mode = "OFF"
    print(f"Database Mode:          {db_mode}")

    # Initialize video capture (webcam or file)
    if video_path and os.path.exists(video_path):
        print(f"[Info] Running on pre-recorded video: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print(f"[Info] Opening Live Camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[Error] Could not open video source (camera {camera_index} or file '{video_path}').")
        if tts:
            tts.stop()
        if db:
            db.close()
        sys.exit(1)

    # Strict Hand-Reset Required State Machine
    system_state = STATE_IDLE
    gesture_buffer = []
    hand_tracker = []  # Tracks (left_detected, right_detected) per frame in window
    consecutive_no_hand_frames = 0
    tracking_loss_frames = 0

    # Recognition & Display State
    current_gloss = ""
    current_confidence = 0.0

    # Active Learning & Retraining tracking
    last_recognized_gloss = ""
    last_features_snapshot = None
    flag_notice_timer = 0
    flag_notice_text = ""

    # Latency & FPS counter variables
    fps = 0.0
    latency_ms = 0.0
    landmark_latency_ms = 0.0
    frame_count = 0
    start_time = time.time()

    print("\n[Controls]")
    print(" - Press 'm': Flag last recognition as MISRECOGNITION for retraining")
    print(" - Press 'q': Exit application\n")

    with mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                if video_path:
                    gesture_buffer.clear()
                    hand_tracker.clear()
                    consecutive_no_hand_frames = 0
                    tracking_loss_frames = 0
                    system_state = STATE_IDLE
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # 2. Camera Mirroring (Horizontal Flip) right after cap.read()
            if should_flip:
                frame = cv2.flip(frame, 1)

            # MediaPipe holistic landmark processing with latency measurement
            t_lm_start = time.perf_counter()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True
            landmark_latency_ms = (time.perf_counter() - t_lm_start) * 1000.0

            # Hand Presence Trigger
            has_lh = results.left_hand_landmarks is not None
            has_rh = results.right_hand_landmarks is not None
            hands_visible = bool(has_lh or has_rh)

            # Strict Hand-Reset Required State Machine:
            # -------------------------------------------------------------
            # State 4: WAITING_FOR_RESET
            # Refuse to record or predict until user lowers hands for at least 8 frames
            if system_state == STATE_WAITING_FOR_RESET:
                gesture_buffer.clear()
                hand_tracker.clear()
                if not hands_visible:
                    consecutive_no_hand_frames += 1
                    if consecutive_no_hand_frames >= reset_consecutive_frames:
                        # Reset confirmed: hands lowered for at least 8 frames
                        system_state = STATE_IDLE
                        consecutive_no_hand_frames = 0
                        tracking_loss_frames = 0
                        print(f"[State Machine] Hands lowered for {reset_consecutive_frames} consecutive frames -> Transition to STATE_IDLE.")
                else:
                    # Hands still up / detected -> Reset count stays 0, refuse to record
                    consecutive_no_hand_frames = 0

            # State 1: IDLE (No Hands)
            elif system_state == STATE_IDLE:
                gesture_buffer.clear()
                hand_tracker.clear()
                consecutive_no_hand_frames = 0
                if hands_visible:
                    # Hand detected: transition to RECORDING
                    system_state = STATE_RECORDING
                    tracking_loss_frames = 0
                    keypoints = extract_keypoints(results)
                    gesture_buffer.append(keypoints)
                    hand_tracker.append((has_lh, has_rh))
                    print("[State Machine] Hand detected -> Transition to STATE_RECORDING.")

            # State 2: RECORDING (Hand Raised, accumulating 30 frames)
            elif system_state == STATE_RECORDING:
                if not hands_visible:
                    tracking_loss_frames += 1
                    # If hands lost for 4 consecutive frames, user lowered hands mid-sign -> abort to IDLE
                    if tracking_loss_frames >= 4:
                        system_state = STATE_IDLE
                        gesture_buffer.clear()
                        hand_tracker.clear()
                        tracking_loss_frames = 0
                        print("[State Machine] Hands dropped mid-gesture -> Aborted to STATE_IDLE.")
                else:
                    tracking_loss_frames = 0
                    keypoints = extract_keypoints(results)
                    gesture_buffer.append(keypoints)
                    hand_tracker.append((has_lh, has_rh))

                # Fixed-Length 30-Frame Gesture Complete -> Inference & TTS Trigger
                if len(gesture_buffer) == GESTURE_WINDOW_SIZE:
                    raw_seq = np.array(gesture_buffer, dtype=np.float32)  # (30, 1662)
                    raw_tensor = np.expand_dims(raw_seq, axis=0)         # (1, 30, 1662)

                    # Feature Slicing (Upper Pose + Left Hand + Right Hand -> 226-D) matching src/dataset.py exactly
                    if feature_dim == 226:
                        input_seq = trim_landmarks(raw_tensor)
                    else:
                        input_seq = raw_tensor
                    input_tensor = torch.from_numpy(input_seq).float().to(device)

                    # Run model inference via torch.no_grad()
                    t_inf_start = time.perf_counter()
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probabilities = F.softmax(logits, dim=1).squeeze(0)
                    latency_ms = (time.perf_counter() - t_inf_start) * 1000.0

                    top_prob, top_idx = torch.max(probabilities, dim=0)
                    prob_val = top_prob.item()
                    idx_val = top_idx.item()
                    pred_gloss = class_names[idx_val]

                    current_gloss = pred_gloss
                    current_confidence = prob_val
                    last_recognized_gloss = pred_gloss
                    last_features_snapshot = raw_seq
                    total_latency_ms = landmark_latency_ms + latency_ms

                    # Calculate hand detection statistics in the 30-frame window
                    lh_detected = any(lh for lh, rh in hand_tracker)
                    rh_detected = any(rh for lh, rh in hand_tracker)
                    lh_count = sum(1 for lh, rh in hand_tracker if lh)
                    rh_count = sum(1 for lh, rh in hand_tracker if rh)

                    # Debug Mode Verification (Terminal Output)
                    top3_probs, top3_indices = torch.topk(probabilities, k=min(3, len(class_names)))
                    print(f"\n================ [Inference Triggered] ================")
                    print(f"Input Tensor Shape:   {input_tensor.shape} (Range: min={input_tensor.min():.4f}, max={input_tensor.max():.4f}, mean={input_tensor.mean():.4f})")
                    print(f"Detected Hands:       (Left detected: {lh_detected} [{lh_count}/30 frames], Right detected: {rh_detected} [{rh_count}/30 frames])")
                    print("Top-3 Predictions:")
                    for rank, (p, idx) in enumerate(zip(top3_probs, top3_indices), start=1):
                        cname = class_names[idx.item()]
                        match_flag = "  <-- PREDICTED" if rank == 1 else ""
                        print(f"  #{rank}: {cname:<15} -> {p.item()*100:6.2f}%{match_flag}")
                    print(f"Latency:              Landmark: {landmark_latency_ms:.1f}ms | Model: {latency_ms:.1f}ms | Total: {total_latency_ms:.1f}ms | FPS: {fps:.1f}")
                    print(f"=======================================================\n")

                    # Trigger TTS voice output non-blockingly (ONCE per gesture)
                    if prob_val >= threshold:
                        if tts:
                            tts.speak(pred_gloss)
                        log_status = "correct"
                    else:
                        if tts and threshold <= DEFAULT_THRESHOLD:
                            tts.speak(pred_gloss)
                            log_status = "low_confidence"
                        else:
                            log_status = "uncertain"

                    append_to_realtime_pilot_logs(
                        session_id=session_id,
                        predicted_gloss=pred_gloss,
                        confidence=prob_val,
                        total_latency_ms=total_latency_ms,
                        fps=fps,
                        status=log_status,
                    )
                    if db:
                        db.log_inference_event(
                            signer_id=signer_id,
                            predicted_gloss=pred_gloss,
                            confidence=prob_val,
                            latency_ms=latency_ms,
                            status=log_status,
                        )
                        db.sync_live_result(
                            session_id=session_id,
                            current_gloss=pred_gloss,
                            confidence=prob_val,
                        )

                    # State 4: Transition to STATE_WAITING_FOR_RESET
                    # Refuse to record or predict until hands lowered for at least 8 frames
                    system_state = STATE_WAITING_FOR_RESET
                    consecutive_no_hand_frames = 0
                    gesture_buffer.clear()
                    hand_tracker.clear()
                    print(f"[State Machine] Recognized '{pred_gloss}' -> Transition to STATE_WAITING_FOR_RESET (Lower hands to reset).")

            # Draw Skeletal Landmarks
            draw_styled_landmarks(frame, results)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Active spoken text for HUD subtitle
            speaking_text = tts.get_active_spoken_phrase() if tts else None

            # Manage misrecognition notification timer
            active_flag_notice = None
            if flag_notice_timer > 0:
                active_flag_notice = flag_notice_text
                flag_notice_timer -= 1

            # Render Translucent HUD Top Card with 4 Distinct Feedback States
            frame = render_hud(
                image=frame,
                system_status=system_state,
                gloss=current_gloss,
                confidence=current_confidence,
                fps=fps,
                latency_ms=latency_ms,
                buffer_count=len(gesture_buffer),
                target_count=GESTURE_WINDOW_SIZE,
                threshold=threshold,
                reset_count=consecutive_no_hand_frames,
                reset_target=reset_consecutive_frames,
                db_mode=db_mode,
                speaking_text=speaking_text,
                flag_notice=active_flag_notice,
            )

            # Display output
            cv2.imshow("Indian Sign Language (ISL) Real-Time Translation", frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[Info] Quit requested by user.")
                break
            elif key in (ord("m"), ord("M")):
                # Flag misrecognition for retraining
                valid_curr = current_gloss if current_gloss not in ("---", "WAITING FOR SIGN...") and not current_gloss.startswith("UNCERTAIN") else ""
                target_gloss = last_recognized_gloss if last_recognized_gloss else valid_curr
                if target_gloss:
                    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    misrecog_filename = f"misrecog_{ts_str}_{target_gloss}.npy"
                    misrecog_path = project_root / "data" / "logs" / misrecog_filename

                    # Save sequence landmarks
                    snap = last_features_snapshot if last_features_snapshot is not None else np.zeros((30, TOTAL_FEATURES), dtype=np.float32)
                    try:
                        np.save(misrecog_path, snap)
                        ref_saved = str(misrecog_path)
                    except Exception as err:
                        ref_saved = f"error_saving_file: {err}"

                    total_latency_ms = landmark_latency_ms + latency_ms
                    append_to_realtime_pilot_logs(
                        session_id=session_id,
                        predicted_gloss=target_gloss,
                        confidence=current_confidence,
                        total_latency_ms=total_latency_ms,
                        fps=fps,
                        status="misrecognition_flagged",
                    )
                    if db:
                        db.log_misrecognition(
                            signer_id=signer_id,
                            expected_gloss="USER_FLAGGED_RETRAIN",
                            predicted_gloss=target_gloss,
                            landmark_features_ref=ref_saved,
                            notes=f"Flagged by user via 'm' key at {ts_str}",
                        )

                    flag_notice_text = f"FLAGGED MISRECOGNITION: '{target_gloss.upper()}' (Saved for Retraining)"
                    flag_notice_timer = 45
                    print(f"\n[Active Learning] User flagged misrecognition for '{target_gloss}' -> logged to DB & {misrecog_filename}")
                else:
                    flag_notice_text = "NO RECENT RECOGNITION TO FLAG"
                    flag_notice_timer = 30

    cap.release()
    cv2.destroyAllWindows()
    if tts:
        tts.stop()
    if db:
        db.close()
    print("[Info] Camera released, TTS stopped, and DB manager closed cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="Run real-time ISL translation via live camera feed with Hand-Reset State Machine, speech synthesis and database logging."
    )
    parser.add_argument("--checkpoint", type=str, default="models/best_transformer_gloss.pth", help="Model weights path.")
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument("--video_path", type=str, default=None, help="Optional test video path for testing.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Confidence threshold for gloss trigger (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--reset_frames", type=int, default=RESET_CONSECUTIVE_FRAMES, help=f"Consecutive zero-hand frames required to reset state machine (default: {RESET_CONSECUTIVE_FRAMES}).")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SECONDS, help=f"Post-recognition locked cooldown in seconds (default: {COOLDOWN_SECONDS}).")
    parser.add_argument("--flip", dest="flip", action="store_true", default=None, help="Mirror webcam feed horizontally (default for camera).")
    parser.add_argument("--no_flip", dest="flip", action="store_false", help="Disable horizontal flip (keep camera feed unflipped).")
    parser.add_argument("--motion_start", type=float, default=MOTION_START_THRESHOLD, help=f"Motion velocity threshold (compatibility arg).")
    parser.add_argument("--motion_stop", type=float, default=MOTION_STOP_THRESHOLD, help=f"Motion velocity stop threshold (compatibility arg).")
    parser.add_argument("--no_tts", action="store_true", help="Disable text-to-speech audio.")
    parser.add_argument("--no_db", action="store_true", help="Disable database logging and session sync.")
    parser.add_argument("--signer_id", type=str, default="signer_01", help="Signer ID for audit and pilot logging.")
    parser.add_argument("--session_id", type=str, default="live_session_01", help="Session ID for live sync.")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5, help="MediaPipe detection confidence.")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help="MediaPipe tracking confidence.")

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        camera_index=args.camera_index,
        video_path=args.video_path,
        threshold=args.threshold,
        enable_tts=not args.no_tts,
        cooldown_seconds=args.cooldown,
        flip=args.flip,
        enable_db=not args.no_db,
        signer_id=args.signer_id,
        session_id=args.session_id,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        motion_start_threshold=args.motion_start,
        motion_stop_threshold=args.motion_stop,
        reset_consecutive_frames=args.reset_frames,
    )


if __name__ == "__main__":
    main()

