"""
extract_landmarks.py
Phase 2 & 3: Indian Sign Language (ISL) Landmark Extraction & Sequence Standardization

Features extracted per frame:
- Pose: 33 landmarks * 4 (x, y, z, visibility) = 132 features
- Face: 468 landmarks * 3 (x, y, z) = 1404 features
- Left Hand: 21 landmarks * 3 (x, y, z) = 63 features
- Right Hand: 21 landmarks * 3 (x, y, z) = 63 features
Total per frame: 1662 features

Sequence Standardization:
- Target sequence length: FIXED_FRAMES = 30
- Uniform downsampling for videos with > 30 frames
- Zero-padding for videos with < 30 frames
- Final sample shape: (30, 1662)
"""

import os
import sys
import warnings

# Suppress verbose C++/absl logs and deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore")

# Ensure Python 3.10 environment is used (which contains mediapipe.solutions)
try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise ImportError("mediapipe.solutions not found in current environment")
except (ImportError, AttributeError):
    conda_py = r"C:\Users\abhis\anaconda3\anaconda\envs\isl_py310\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        result = subprocess.run([conda_py] + sys.argv)
        sys.exit(result.returncode)
    raise

import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# Constants
POSE_LANDMARKS = 33
POSE_FEATURES = POSE_LANDMARKS * 4  # x, y, z, visibility -> 132
FACE_LANDMARKS = 468
FACE_FEATURES = FACE_LANDMARKS * 3  # x, y, z -> 1404
HAND_LANDMARKS = 21
HAND_FEATURES = HAND_LANDMARKS * 3  # x, y, z -> 63 each (left and right)
TOTAL_FEATURES = POSE_FEATURES + FACE_FEATURES + (HAND_FEATURES * 2)  # 1662
FIXED_FRAMES = 30

mp_holistic = mp.solutions.holistic

_worker_holistic = None


def _init_worker(min_detection_confidence: float, min_tracking_confidence: float):
    """Initializes a separate MediaPipe Holistic model instance per worker process."""
    global _worker_holistic
    import os, warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "2"
    warnings.filterwarnings("ignore")
    import mediapipe as mp
    _worker_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_keypoints_from_results(results) -> np.ndarray:
    """
    Extracts and flattens landmarks from MediaPipe Holistic results.
    Returns a 1D numpy array of shape (TOTAL_FEATURES,) = (1662,).
    """
    # Pose: 33 keypoints (x, y, z, visibility) -> 132 features
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_FEATURES, dtype=np.float32)

    # Face: 468 keypoints (x, y, z) -> 1404 features
    if results.face_landmarks:
        # Guarantee exactly 468 landmarks
        face_lms = results.face_landmarks.landmark[:FACE_LANDMARKS]
        face = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_lms],
            dtype=np.float32,
        ).flatten()
        if len(face) < FACE_FEATURES:
            face = np.pad(face, (0, FACE_FEATURES - len(face)), mode="constant")
    else:
        face = np.zeros(FACE_FEATURES, dtype=np.float32)

    # Left Hand: 21 keypoints (x, y, z) -> 63 features
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        lh = np.zeros(HAND_FEATURES, dtype=np.float32)

    # Right Hand: 21 keypoints (x, y, z) -> 63 features
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        rh = np.zeros(HAND_FEATURES, dtype=np.float32)

    # Concatenate all into a single 1D feature vector
    return np.concatenate([pose, face, lh, rh], axis=0).astype(np.float32)


def process_video(video_path: str, holistic) -> np.ndarray:
    """
    Reads a video file, extracts holistic landmarks per frame,
    and standardizes the sequence to shape (FIXED_FRAMES, TOTAL_FEATURES).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n[Warning] Could not open video: {video_path}")
        return np.zeros((FIXED_FRAMES, TOTAL_FEATURES), dtype=np.float32)

    frame_features_list = []

    # Reset holistic tracking state between videos if supported
    if hasattr(holistic, "reset"):
        holistic.reset()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        keypoints = extract_keypoints_from_results(results)
        frame_features_list.append(keypoints)

    cap.release()

    total_extracted = len(frame_features_list)

    if total_extracted == 0:
        print(f"\n[Warning] No frames decoded from: {video_path}")
        return np.zeros((FIXED_FRAMES, TOTAL_FEATURES), dtype=np.float32)

    raw_sequence = np.array(frame_features_list, dtype=np.float32)

    # Sequence Standardization:
    if total_extracted > FIXED_FRAMES:
        # Uniform downsampling
        indices = np.linspace(0, total_extracted - 1, FIXED_FRAMES, dtype=int)
        standardized_sequence = raw_sequence[indices]
    elif total_extracted < FIXED_FRAMES:
        # Tail zero-padding
        padding = np.zeros((FIXED_FRAMES - total_extracted, TOTAL_FEATURES), dtype=np.float32)
        standardized_sequence = np.vstack([raw_sequence, padding])
    else:
        standardized_sequence = raw_sequence

    return standardized_sequence


def _process_video_task(task_tuple):
    """Worker task function executed in parallel processes."""
    cls_name, video_file_str, out_npy_file_str, overwrite = task_tuple
    out_npy_file = Path(out_npy_file_str)
    if out_npy_file.exists() and not overwrite:
        return cls_name, video_file_str, "skipped"

    global _worker_holistic
    try:
        features_sequence = process_video(video_file_str, _worker_holistic)
        np.save(str(out_npy_file), features_sequence)
        return cls_name, video_file_str, "saved"
    except Exception as e:
        return cls_name, video_file_str, f"error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks and standardize sequence lengths for ISL videos."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/raw_videos",
        help="Path to directory containing class folders of raw mp4 videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_data",
        help="Path to directory where processed .npy files will be saved.",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default="data/label_map.json",
        help="Path where label map JSON file will be saved.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe min detection confidence.",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="MediaPipe min tracking confidence.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel worker processes (default: 6).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files if present.",
    )

    args = parser.parse_args()

    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)
    label_map_path = Path(args.label_map)

    # Fallback to custom_dataset if default data/raw_videos is not found
    if not source_path.exists():
        fallback_path = Path("custom_dataset")
        if fallback_path.exists():
            print(f"[Info] '{source_path}' not found, falling back to '{fallback_path}'.")
            source_path = fallback_path
        else:
            print(f"[Error] Source directory '{source_path}' does not exist.")
            sys.exit(1)

    print(f"=== ISL Holistic Landmark Extraction & Standardization ===")
    print(f"Source Directory: {source_path.resolve()}")
    print(f"Output Directory: {output_path.resolve()}")
    print(f"Label Map Output: {label_map_path.resolve()}")
    print(f"Fixed Frames:     {FIXED_FRAMES}")
    print(f"Total Features:   {TOTAL_FEATURES} (Pose: {POSE_FEATURES}, Face: {FACE_FEATURES}, LH: {HAND_FEATURES}, RH: {HAND_FEATURES})")
    print(f"Workers:          {args.workers}")

    # Discover classes (only subdirectories)
    class_names = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    if not class_names:
        print(f"[Error] No class directories found in '{source_path}'.")
        sys.exit(1)

    # Generate and save label map
    label_map = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)
    print(f"\n[Info] Saved label map with {len(label_map)} classes to '{label_map_path}'.")

    # Collect all video tasks
    video_tasks = []
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for cls_name in class_names:
        cls_dir = source_path / cls_name
        out_cls_dir = output_path / cls_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        for video_file in sorted(cls_dir.iterdir()):
            if video_file.is_file() and video_file.suffix.lower() in valid_extensions:
                sample_name = video_file.stem
                out_npy_file = out_cls_dir / f"{sample_name}.npy"
                video_tasks.append((cls_name, str(video_file), str(out_npy_file), args.overwrite))

    total_videos = len(video_tasks)
    print(f"[Info] Found {total_videos} videos across {len(class_names)} classes.")

    if total_videos == 0:
        print("[Warning] No video files found to process.")
        sys.exit(0)

    saved_count = 0
    skipped_count = 0
    error_count = 0

    if args.workers <= 1:
        # Single-process mode
        print(f"\nInitializing single-process MediaPipe Holistic model...")
        with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as holistic:
            pbar = tqdm(video_tasks, desc="Extracting Landmarks", unit="video")
            for cls_name, video_file_str, out_npy_str, overwrite in pbar:
                video_file = Path(video_file_str)
                out_npy = Path(out_npy_str)
                pbar.set_postfix(cls=cls_name, file=video_file.name[:18])

                if out_npy.exists() and not overwrite:
                    skipped_count += 1
                    continue

                try:
                    features_sequence = process_video(video_file_str, holistic)
                    np.save(str(out_npy), features_sequence)
                    saved_count += 1
                except Exception as e:
                    print(f"\n[Error] Failed processing {video_file}: {e}")
                    error_count += 1
    else:
        # Multi-process parallel mode
        print(f"\nStarting {args.workers} parallel worker processes with MediaPipe Holistic...")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.min_detection_confidence, args.min_tracking_confidence),
        ) as executor:
            pbar = tqdm(total=total_videos, desc="Extracting Landmarks", unit="video")
            futures = [executor.submit(_process_video_task, task) for task in video_tasks]

            for future in as_completed(futures):
                cls_name, video_file_str, status = future.result()
                video_name = Path(video_file_str).name
                pbar.set_postfix(cls=cls_name, file=video_name[:18], status=status)
                pbar.update(1)

                if status == "saved":
                    saved_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1
                    print(f"\n[Error] {video_name}: {status}")

            pbar.close()

    # Summary Statistics
    total_npy = sum(len(list((output_path / cls).glob("*.npy"))) for cls in class_names)
    print("\n" + "=" * 55)
    print("           EXTRACTION SUMMARY STATISTICS")
    print("=" * 55)
    print(f"Total Classes:               {len(class_names)}")
    print(f"Total Videos Processed:      {saved_count}")
    print(f"Total Videos Skipped:        {skipped_count}")
    print(f"Total Errors:                {error_count}")
    print(f"Total .npy Files Generated:  {total_npy}")
    print(f"Feature Vector Shape:        ({FIXED_FRAMES}, {TOTAL_FEATURES})")
    print(f"Processed Dataset Root:      {output_path.resolve()}")
    print(f"Label Map:                   {label_map_path.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
