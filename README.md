# 🤟 Indian Sign Language (ISL) Translation System
### End-to-End Deep Learning, Speech Synthesis & Cloud Architecture

An accessible, real-time AI translation system that converts continuous Indian Sign Language (ISL) gestures into conversational English text and spoken speech. Built with **MediaPipe Holistic**, **PyTorch Transformer Encoders**, **pyttsx3 Offline TTS**, **MongoDB Atlas**, **Firebase Firestore**, and **Streamlit**.

---

## 🌟 Key Features

1. **Targeted Salient Keypoint Extraction & Trimming (MediaPipe Holistic)**:
   - Extracts MediaPipe landmarks and slices down to **226 salient features** per frame: Upper Pose (pts 0 to 24 / 100 values), Left Hand (21 pts / 63 values), and Right Hand (21 pts / 63 values).
   - Completely drops 1404 noisy face mesh features, resolving the curse of dimensionality on small-to-medium datasets.
   - Standardized 30-frame sequence windows $(30, 226)$.

2. **Regularized Transformer Sequence Classifier**:
   - Sinusoidal Positional Encoding preserving temporal frame order across the 30-frame sequence.
   - 2-Layer Transformer Encoder with Pre-LayerNorm, 4 multi-head self-attentions ($d_{model} = 128$, $d_{ff} = 256$, dropout = 0.35).
   - Linear projection LayerNorm and regularized classification head.
   - **305,061 parameters** (85.1% reduction from original 2.04M), achieving **95.33% validation accuracy** and **91.59% test accuracy**.

3. **Low-Latency Real-Time Desktop Inference (OpenCV)**:
   - Sliding-window temporal buffer (`collections.deque(maxlen=30)`).
   - Ultra-fast forward pass: **~0.78 ms** inference latency on CPU (**~31 FPS** total pipeline throughput).
   - Translucent HUD overlay displaying live gloss, confidence progress bar, FPS, inference latency, and skeletal landmarks.

4. **Speech & Conversational Synthesis (TTS)**:
   - Non-blocking daemon worker thread consuming an internal speech queue.
   - Maps isolated signs into natural conversational English (e.g., `"atm"` ➡️ `"Where is the nearest ATM?"`).
   - SAPI5 offline engine with configurable debouncing and cooldown.

5. **Cloud Database & Active Learning Architecture**:
   - **MongoDB Atlas**: Asynchronously stores vocabulary metadata, model metrics, live inference event logs, and user-flagged misrecognitions.
   - **Firebase Firestore**: Real-time sync for external mobile or web app clients.
   - **Zero-Crash Local Fallback**: Automatically redirects audit logs to `data/logs/` if cloud credentials are not supplied.

6. **Interactive Pilot Web Demo (Gradio 6)**:
   - Web application supporting video upload, live webcam recording, and instant sample tests.
   - Visual audio player, top-5 confidence gauge, and community feedback form for active learning retraining.

---

## 📁 Repository Structure

```text
isl_project/
│
├── .env.example                     # Environment template (MongoDB & Firebase)
├── app.py                           # Gradio 6 Pilot Web Demo & Feedback UI
├── README.md                        # Complete Project Documentation
│
├── custom_dataset/                  # Raw dataset directory (or data/raw_videos)
│   ├── atm/*.mp4
│   ├── namaste/*.mp4
│   └── ... (37 gloss classes)
│
├── data/
│   ├── label_map.json               # 37-class mapping dictionary
│   ├── processed_data/              # Extracted landmark sequences (30, 1662)
│   └── logs/                        # Local fallback JSONL logs & misrecognition .npy files
│
├── models/
│   └── best_transformer_gloss.pth   # Best trained Transformer weights (Val Acc: 95.33%, Test Top-1: 91.59%)
│
├── reports/
│   ├── benchmark_summary.json       # Consolidated latency and accuracy metrics
│   ├── classification_report.txt   # Precision, recall, and f1-score per gloss
│   ├── confusion_matrix.png         # High-resolution normalized confusion matrix
│   ├── confusion_pairs.csv          # Tableau-ready pairwise confusion matrix records (37x37)
│   ├── model_evaluation_metrics.csv # Tableau-ready per-gloss precision, recall, f1, accuracy
│   ├── realtime_pilot_logs.csv      # Tableau-ready live & pilot inference stream records
│   ├── test_metrics.json            # Held-out test evaluation results
│   └── training_curves.png          # Epoch-by-epoch loss and accuracy curves
│
└── src/
    ├── benchmark.py                 # Benchmarking and latency profiling script
    ├── dataset.py                   # PyTorch ISLDataset with augmentations
    ├── db_manager.py                # Database manager (MongoDB Atlas + Firebase + Tableau CSV)
    ├── export_tableau_data.py       # Exporter for Tableau Desktop / Public CSV datasets
    ├── extract_landmarks.py         # MediaPipe Holistic feature extraction pipeline
    ├── model.py                     # Transformer Sequence Classifier architecture
    ├── realtime_inference.py        # Real-time OpenCV live camera translation pipeline
    ├── train.py                     # Training loop with Cosine Annealing and early stopping
    └── tts_service.py               # Asynchronous Speech Synthesis engine
```

---

## 📈 Tableau Business Intelligence & Analytics Integration

All model evaluations and real-time pilot logs are formatted as clean, schema-consistent CSV files ready to be dragged directly into **Tableau Desktop**, **Tableau Prep**, or **Tableau Public**:

| Dataset File | Key Columns | Tableau Visualization Use Case |
| :--- | :--- | :--- |
| **`reports/model_evaluation_metrics.csv`** | `gloss_name`, `precision`, `recall`, `f1_score`, `support_count`, `top1_accuracy` | Bar charts, radar charts, and scatter plots comparing per-sign accuracy vs sample support. |
| **`reports/confusion_pairs.csv`** | `actual_gloss`, `predicted_gloss`, `count`, `normalized_percentage` | Interactive Confusion Matrix Heatmap with color intensity on `normalized_percentage`. |
| **`reports/realtime_pilot_logs.csv`** | `timestamp`, `session_id`, `predicted_gloss`, `confidence`, `total_latency_ms`, `fps`, `status` | Real-time session monitoring dashboard: latency over time, FPS throughput, and misrecognition rate by signer. |

To regenerate or refresh the evaluation CSVs at any time:
```bash
python src/export_tableau_data.py
```
*(Also automatically refreshed whenever `python src/benchmark.py` is run).*

---

## ⚙️ Environment Setup & Prerequisites

### 1. Requirements
- **Python**: 3.10 (Recommended: Conda environment `isl_py310`)
- **PyTorch**: 2.1+
- **MediaPipe**: 0.10+
- **OpenCV**: 4.8+
- **Gradio**: 6.29+
- **Pymongo & Python-dotenv**: For database connectivity

### 2. Environment Variables (.env)
Copy `.env.example` to `.env` in the project root:
```bash
cp .env.example .env
```
Configure your credentials:
```env
# MongoDB Atlas (Optional - falls back to data/logs/ if empty)
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGO_DB_NAME=isl_database

# Firebase Admin SDK (Optional)
FIREBASE_CREDENTIALS_PATH=config/firebase_credentials.json

# Pilot Defaults
DEFAULT_SIGNER_ID=signer_pilot_01
DEFAULT_SESSION_ID=live_session_01
```

---

## 🚀 Execution Guide

### Step 1: Extract MediaPipe Landmarks
Processes raw `.mp4` videos into standardized `(30, 1662)` float32 arrays and generates `data/label_map.json`:
```bash
python src/extract_landmarks.py --raw_dir custom_dataset --output_dir data/processed_data --target_len 30
```

### Step 2: Train the Transformer Classifier
Trains the sequence classifier with temporal augmentations, Cosine Annealing LR schedule, label smoothing, and evaluation on a held-out test split:
```bash
python src/train.py --epochs 60 --batch_size 16 --lr 1e-3 --patience 12
```
*Outputs: `models/best_transformer_gloss.pth`, `reports/confusion_matrix.png`, `reports/training_curves.png`.*

### Step 3: Run the Comprehensive Benchmark & Profiler
Evaluates the test split and measures latency breakdown across all pipeline modules:
```bash
python src/benchmark.py
```
*Outputs: Consolidated metrics exported to `reports/benchmark_summary.json`.*

### Step 4: Real-Time Live Camera Inference & Test Verification
Launches real-time camera translation with Hand Presence Windowing, speech synthesis, and live debug output:
```bash
# Live webcam translation (Hand Presence Windowing & 30-frame capture)
python src/realtime_inference.py --camera_index 0 --threshold 0.70 --cooldown 1.5

# Optional: Disable horizontal flip if using a rear/unmirrored camera:
python src/realtime_inference.py --no_flip

# Pre-recorded video test verification (verifies model weights on ground-truth samples):
python src/test_video_sample.py --video data/raw_videos/namaste/namaste_1_001.mp4
```
**Desktop Controls & Features**:
- **Hand Presence Trigger**: Automatically enters `"IDLE: SHOW HAND TO SIGN"` when hands are resting/lowered, completely eliminating zero-padding hallucinations.
- **Fixed-Length 30-Frame Capture**: Automatically records 30 frames upon hand detection with a visual progress bar and runs inference immediately.
- **Horizontal Mirroring**: Automatically applies `cv2.flip(frame, 1)` so physical hand movements intuitively match camera coordinates.
- **Debug Mode Terminal Output**: Prints input tensor shape, numerical range, active hand presence (left/right), and Top-3 predictions with confidence scores on every inference.
- **1.5s Result Lock**: Locks the recognized gloss on screen and plays non-blocking TTS audio, giving the user ample time to lower their hands.
- `'m'`: Flag the current sign as a **misrecognition** (stores the 30-frame `.npy` landmark snapshot and pushes an active learning retraining ticket to MongoDB).
- `'q'`: Exit cleanly.

### Step 5: Launch the Interactive Pilot Web UI (Streamlit)
Starts the accessible, modern Streamlit browser application on `http://localhost:8501`:
```bash
streamlit run app.py
```
**Web Features**:
- **Tab 1: Live Camera Translation**: Continuous 30-frame windowing with strict Hand-Reset state machine, custom CSS cards, animated confidence bars, and offline TTS voice playback.
- **Tab 2: Video Verification (Demo)**: Built-in library of 9 pre-loaded sign clips (`namaste`, `atm`, `Nice`, `home`, `bank`, `kitchen`, etc.) plus custom `.mp4`/`.avi` upload for failsafe project evaluation.
- **Tab 3: Model Performance Visualizer**: Audited normalized confusion matrix (`reports/confusion_matrix.png`), training curves (`reports/training_curves.png`), and per-gloss precision/recall table.
- **Tab 4: Pilot Feedback & Active Learning**: Community misrecognition reporting form connected directly to MongoDB Atlas and `reports/realtime_pilot_logs.csv`.

---

## 📊 Performance Benchmarks (Held-Out Test Set)

| Metric | Baseline (1662-D Dense) | Optimized (226-D Trimmed) | Improvement |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 34.58% | **92.52%** (99/107) | **+57.94%** |
| **Top-5 Accuracy** | 73.83% | **98.13%** (105/107) | **+24.30%** |
| **Macro F1-Score** | 0.2838 | **0.9168** | **+0.6330** |
| **Weighted F1-Score** | 0.2894 | **0.9179** | **+0.6285** |
| **Validation Accuracy (Best)** | 37.38% | **95.33%** | **+57.95%** |
| **Parameter Count** | 2,044,965 | **305,061** | **-85.1%** |
| **Model Checkpoint Size** | 24.7 MB | **3.7 MB** | **-85.0%** |
| **MediaPipe Frame Extraction** | 32.5 ms | **31.65 ms** | — |
| **Transformer Forward Pass** | 1.61 ms | **0.78 ms** | **2.06x faster** |
| **TTS Speech Trigger Overhead** | < 0.001 ms | **< 0.001 ms** (Non-blocking) | — |
| **Total Incremental Latency** | 34.1 ms | **32.43 ms** | — |
| **Continuous Throughput** | 29.3 FPS | **30.8 FPS** | **Real-time capable** |

---

## 🔄 Active Learning & Retraining Pipeline

1. **Capture**: When a user flags a misrecognition (via key `'m'` in OpenCV or the web UI form), the exact 30-frame MediaPipe landmark sequence is saved to `data/logs/misrecog_<timestamp>_<gloss>.npy`.
2. **Audit & Log**: The event record (Signer ID, predicted gloss, expected gloss, landmark reference path, timestamp) is asynchronously committed to MongoDB Atlas under collection `misrecognitions`.
3. **Fine-Tuning**: Retraining scripts read pending misrecognition logs from MongoDB Atlas, merge corrected landmark arrays into `data/processed_data/`, and re-trigger model training.
