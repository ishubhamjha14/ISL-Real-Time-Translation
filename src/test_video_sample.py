"""
test_video_sample.py
Verification script to test the trained ISL Transformer model on raw pre-recorded videos.
Evaluates landmark extraction, feature trimming (226-D), and model predictions.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset import trim_landmarks
from src.realtime_inference import load_isl_model, extract_keypoints, mp_holistic

def test_video(video_path: str, checkpoint_path: str = "models/best_transformer_gloss.pth", flip: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, feature_dim = load_isl_model(checkpoint_path, device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    
    expected_class = Path(video_path).parent.name
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=======================================================")
    print(f"Testing Video:        {video_path}")
    print(f"Ground Truth Gloss:   '{expected_class}'")
    print(f"Total Video Frames:   {total_frames}")
    print(f"Horizontal Flip:      {flip}")
    print(f"=======================================================")
    
    frame_features = []
    left_hand_detected_count = 0
    right_hand_detected_count = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            if flip:
                frame = cv2.flip(frame, 1)
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            
            if results.left_hand_landmarks:
                left_hand_detected_count += 1
            if results.right_hand_landmarks:
                right_hand_detected_count += 1
                
            kp = extract_keypoints(results)
            frame_features.append(kp)
            
    cap.release()
    
    if len(frame_features) == 0:
        print("[Error] No frames were extracted from video.")
        return
        
    print(f"Extracted {len(frame_features)} frames.")
    print(f"Hand detections -> Left Hand: {left_hand_detected_count}/{len(frame_features)} frames | Right Hand: {right_hand_detected_count}/{len(frame_features)} frames")
    
    # Resample to exactly 30 frames using linear interpolation
    raw_arr = np.array(frame_features, dtype=np.float32)
    indices = np.linspace(0, len(raw_arr) - 1, 30, dtype=int)
    resampled_arr = raw_arr[indices] # (30, 1662)
    
    raw_tensor = np.expand_dims(resampled_arr, axis=0) # (1, 30, 1662)
    trimmed_tensor = trim_landmarks(raw_tensor) if feature_dim == 226 else raw_tensor
    input_tensor = torch.from_numpy(trimmed_tensor).float().to(device)
    
    print(f"Input tensor shape:   {input_tensor.shape} (Expected: (1, 30, {feature_dim}))")
    print(f"Input tensor range:   min={input_tensor.min():.4f}, max={input_tensor.max():.4f}, mean={input_tensor.mean():.4f}")
    
    # Run Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        
    top_probs, top_indices = torch.topk(probs, k=min(5, len(class_names)))
    
    print("\nTop Predictions:")
    matched = False
    for rank, (p, idx) in enumerate(zip(top_probs, top_indices), start=1):
        cname = class_names[idx.item()]
        match_marker = " [MATCH!]" if cname.lower() == expected_class.lower() else ""
        if rank == 1 and cname.lower() == expected_class.lower():
            matched = True
        print(f"  #{rank}: {cname:<15} -> {p.item()*100:6.2f}%{match_marker}")
        
    print("-------------------------------------------------------")
    if matched:
        print(f"SUCCESS: Ground truth '{expected_class}' correctly recognized as Top-1 with {top_probs[0].item()*100:.2f}% confidence!")
    else:
        print(f"MISMATCH: Expected '{expected_class}', but predicted '{class_names[top_indices[0].item()]}'.")
    print("=======================================================\n")
    return matched

def main():
    parser = argparse.ArgumentParser(description="Test pre-recorded video samples against ISL classifier.")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (.mp4).")
    parser.add_argument("--checkpoint", type=str, default="models/best_transformer_gloss.pth", help="Model weights.")
    parser.add_argument("--flip", action="store_true", help="Apply horizontal flip (cv2.flip(frame, 1)).")
    args = parser.parse_args()
    
    if args.video:
        test_video(args.video, args.checkpoint, flip=args.flip)
    else:
        # Test on a few diverse samples from data/raw_videos
        candidates = [
            "data/raw_videos/atm/atm_1_001.mp4",
            "data/raw_videos/namaste/namaste_1_001.mp4",
            "data/raw_videos/Nice/Nice_2_001.mp4",
            "data/raw_videos/home/home_2_001.mp4",
            "data/raw_videos/bank/bank_2_001.mp4",
        ]
        tested = 0
        passed = 0
        for cand in candidates:
            if os.path.exists(cand):
                success = test_video(cand, args.checkpoint, flip=args.flip)
                tested += 1
                if success:
                    passed += 1
        if tested > 0:
            print(f"Summary: {passed}/{tested} sample videos passed Top-1 correctly ({passed/tested*100:.1f}%).")

if __name__ == "__main__":
    main()
