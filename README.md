# MoodLens — Multimodal Video Emotion & Sentiment Analysis

MoodLens is an end-to-end multimodal deep learning system and full-stack web application that analyzes spoken utterances in video files across **textual**, **visual**, and **acoustic** modalities to predict fine-grained **7-class emotions** and **3-class sentiment polarities**.

Built around a **PyTorch Late-Fusion** architecture trained on the **MELD** (Multimodal EmotionLines Dataset), MoodLens pairs a **FastAPI** inference and analytics backend (`backend/`) with a **Next.js + shadcn/ui** web client (`frontend/`).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Multimodal Neural Network (`MultimodalSentimentModel`)](#multimodal-neural-network-multimodalsentimentmodel)
3. [Training Strategy & Class Imbalance Mitigation](#training-strategy--class-imbalance-mitigation)
4. [Inference, Video Chunking & Valence Analytics](#inference-video-chunking--valence-analytics)
5. [FastAPI REST API Reference](#fastapi-rest-api-reference)
6. [Repository Directory Structure](#repository-directory-structure)
7. [Prerequisites & Local Development Setup](#prerequisites--local-development-setup)
8. [Environment Variables Reference](#environment-variables-reference)
9. [Evaluation, Profiling, Ablation & Error Analysis Suite](#evaluation-profiling-ablation--error-analysis-suite)
10. [Automated Unit & Integrity Testing](#automated-unit--integrity-testing)
11. [Troubleshooting](#troubleshooting)

---

## System Architecture

### 1. End-to-End Processing Pipeline

```text
+-----------------------------------------------------------------------------------+
|                        Next.js + shadcn/ui Frontend (Port 3000)                   |
|   - Multipart video upload (MP4 / MOV / WebM / AVI)                               |
|   - Per-utterance emotion & sentiment probability visualization                   |
+-----------------------------------------+-----------------------------------------+
                                          | POST /analyze/video (multipart/form-data)
                                          v
+-----------------------------------------------------------------------------------+
|                          FastAPI Backend Service (Port 8000)                      |
|                                                                                   |
|  1. Media Validation & Chunking (deployment/video_chunker.py)                     |
|     - Probes duration via ffprobe; splits long videos into bounded sub-clips      |
|                                                                                   |
|  2. Speech Transcription & Utterance Segmentation (deployment/inference.py)       |
|     - Extracts audio track via FFmpeg (16 kHz)                                    |
|     - Transcribes speech with Whisper to obtain [start_time, end_time, text]      |
|                                                                                   |
|  3. Multimodal Tensor Extraction (Per Utterance)                                  |
|     - Text  -> bert-base-uncased tokenizer  -> input_ids & attention_mask (128)   |
|     - Video -> 30 uniform RGB frames        -> (30, 3, 224, 224)                  |
|     - Audio -> 64-band Mel-Spectrogram      -> (1, 64, 300)                       |
|                                                                                   |
|  4. Late-Fusion PyTorch Inference (MultimodalSentimentModel)                      |
|     - TextEncoder (128d) || VideoEncoder (128d) || AudioEncoder (128d)            |
|     - FusionLayer (384d -> 256d) -> Emotion Head (7) & Sentiment Head (3)         |
|                                                                                   |
|  5. Post-Processing & Analytics (deployment/analytics_aggregator.py)              |
|     - Computes valence trajectory, mood-shift timestamps & CSV/JSON summaries     |
+-----------------------------------------------------------------------------------+
```

### 2. Full-Stack Component Responsibilities

| Layer | Stack | Directory | Primary Responsibilities |
| :--- | :--- | :--- | :--- |
| **ML Core & Training** | PyTorch, `transformers`, `torchvision`, `torchaudio`, `scikit-learn` | `backend/src/training/` | MELD preprocessing, multimodal encoders, weighted multi-task training, per-class metrics, ablation, dataset profiling, error analysis |
| **Inference & API** | FastAPI, Uvicorn, OpenAI Whisper, FFmpeg | `backend/src/deployment/` | Model lifecycle management, utterance slicing, 60s video chunking, REST endpoints, rate limiting, valence analytics |
| **Test Suite** | Python `unittest` | `backend/tests/` | Tensor shape verification, forward-pass NaN checks, chunk offset merging, offline analytics & profiler unit tests |
| **Web Client** | Next.js (App Router), React, TypeScript, Tailwind CSS, `shadcn/ui` | `frontend/` | Client-side file validation, upload state management, probability bars & emoji sentiment/emotion indicators |

---

## Multimodal Neural Network (`MultimodalSentimentModel`)

### 1. Why Late Fusion?

Raw video, audio waveforms, and tokenized text operate at incompatible temporal resolutions and dimensionalities (5D spatiotemporal tensors vs. 2D spectrograms vs. discrete token sequences). Rather than concatenating raw inputs (Early Fusion), MoodLens encodes each modality independently using specialized pretrained or convolutional backbones into a unified **128-dimensional latent space** before fusing them.

### 2. Encoder & Fusion Layer Specifications

| Module | Backbone / Layer Architecture | Input Shape | Output Shape | Trainable Status |
| :--- | :--- | :--- | :--- | :--- |
| **`TextEncoder`** | `BertModel.from_pretrained("bert-base-uncased")` (`pooler_output` `[CLS]`) $\rightarrow$ `Linear(768, 128)` | `input_ids`: `(B, 128)`<br>`attention_mask`: `(B, 128)` | `(B, 128)` | BERT frozen; `Linear(768, 128)` trainable (`lr=8e-6`) |
| **`VideoEncoder`** | `torchvision.models.video.r3d_18(pretrained=True)` (transposes `(B, T, C, H, W)` $\rightarrow$ `(B, C, T, H, W)`) $\rightarrow$ `Linear(512, 128)` + `ReLU` + `Dropout(0.2)` | `(B, 30, 3, 224, 224)` | `(B, 128)` | 3D ResNet-18 frozen; `fc` head trainable (`lr=8e-5`) |
| **`AudioEncoder`** | `Conv1d(64, 64, k=3)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` $\rightarrow$ `Conv1d(64, 128, k=3)` $\rightarrow$ `BatchNorm1d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `AdaptiveAvgPool1d(1)` $\rightarrow$ `Linear(128, 128)` + `ReLU` + `Dropout(0.2)` | `(B, 1, 64, 300)` | `(B, 128)` | Conv blocks frozen; `projection` head trainable (`lr=8e-5`) |
| **`FusionLayer`** | `torch.cat([text, video, audio], dim=1)` $\rightarrow$ `Linear(384, 256)` $\rightarrow$ `BatchNorm1d(256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` | `(B, 384)` | `(B, 256)` | Fully trainable (`lr=5e-4`) |
| **`emotion_classifier`** | `Linear(256, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(64, 7)` | `(B, 256)` | `(B, 7)` | Fully trainable (`lr=5e-4`) |
| **`sentiment_classifier`** | `Linear(256, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(64, 3)` | `(B, 256)` | `(B, 3)` | Fully trainable (`lr=5e-4`) |

### 3. Target Label Mappings & Cross-Head Polarity

MoodLens performs **multi-task classification**, predicting both a discrete emotion and a coarse sentiment polarity for every utterance:

* **Emotion Classes (`7`):**
  * `0`: `anger` (Negative polarity)
  * `1`: `disgust` (Negative polarity)
  * `2`: `fear` (Negative polarity)
  * `3`: `joy` (Positive polarity)
  * `4`: `neutral` (Neutral polarity)
  * `5`: `sadness` (Negative polarity)
  * `6`: `surprise` (Context-dependent: Negative, Neutral, or Positive)
* **Sentiment Classes (`3`):**
  * `0`: `negative`
  * `1`: `neutral`
  * `2`: `positive`

---

## Training Strategy & Class Imbalance Mitigation

### 1. Multi-Task Objective Function

The network is trained jointly by summing two label-smoothed, class-weighted Cross-Entropy losses:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}\!\left(\hat{y}_{\text{emotion}},\, y_{\text{emotion}};\, w_{\text{emotion}},\, \epsilon=0.05\right) + \mathcal{L}_{\text{CE}}\!\left(\hat{y}_{\text{sentiment}},\, y_{\text{sentiment}};\, w_{\text{sentiment}},\, \epsilon=0.05\right)$$

### 2. Handling MELD Class Skew

In the MELD dataset, `neutral` utterances account for roughly ~47% of all samples, while minority classes such as `fear` and `disgust` represent less than 3% each. Without reweighting, models collapse toward predicting `neutral`. `compute_class_weights()` in `backend/src/training/models.py` computes normalized inverse-frequency weights across valid training samples:

$$w_c = \frac{1 / N_c}{\sum_{k=1}^{C} (1 / N_k)}$$

### 3. Optimization & Regularization Configuration

* **Optimizer:** `Adam` with `weight_decay=1e-5` and parameter-group learning rates (`8e-6` for text projection, `8e-5` for audio/video heads, `5e-4` for fusion and classifiers).
* **LR Scheduler:** `ReduceLROnPlateau(mode="min", factor=0.1, patience=2)` monitored on validation total loss.
* **Stability & Overfitting Controls:** Gradient norm clipping (`max_norm=1.0`), `BatchNorm1d` + `Dropout(0.2–0.3)` in projection/fusion layers, and `label_smoothing=0.05`.
* **Telemetry:** Real-time scalar tracking via PyTorch `SummaryWriter` (TensorBoard) logging train/val losses, weighted precision, and accuracy per step/epoch.

---

## Inference, Video Chunking & Valence Analytics

### 1. Long-Video Chunking (`backend/src/deployment/video_chunker.py`)

Long videos can exceed single-request inference budgets or memory limits. `video_chunker.py` provides:
* `get_video_duration(video_path)`: Queries exact container duration via `ffprobe`.
* `split_video_into_chunks(video_path, output_dir, max_chunk_duration=60.0)`: Passes clips $\le 60\text{s}$ through unchanged and splits longer videos into sequential sub-clips (`libx264` + `aac`) with exact `start_offset` metadata.
* `merge_chunk_predictions(chunk_results)`: Shifts every utterance's `start` and `end` timestamps by its parent chunk's `start_offset` and aggregates overall emotion/sentiment distributions.

### 2. Valence Trajectory & Mood-Shift Detection (`backend/src/deployment/analytics_aggregator.py`)

`analytics_aggregator.py` transforms raw utterance probability lists into structured video-level insights:
* **Continuous Valence Score:** Computes $v_i = P(\text{positive}) - P(\text{negative}) \in [-1.0, +1.0]$ per utterance.
* **Duration-Weighted Distributions:** Weighs each predicted emotion by utterance duration ($t_{\text{end}} - t_{\text{start}}$) in addition to raw utterance counts.
* **Mood-Shift Detection (`detect_mood_shifts`):** Flags timestamps where high-confidence consecutive utterances transition across emotion or sentiment boundaries along with the `valence_delta`.
* **Artifact Export:** Generates tabular utterance reports via `export_utterances_csv()` and full analytical summaries via `export_analytics_json()`.

---

## FastAPI REST API Reference

### Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns service health and whether `MultimodalSentimentModel` weights are loaded in memory. |
| `POST` | `/analyze/video` | Accepts a multipart video file upload (`video` field) and returns utterance-level emotion and sentiment probability rankings. |

### Sample Response (`POST /analyze/video`)

```json
{
  "utterances": [
    {
      "start_time": 0.0,
      "end_time": 1.6,
      "text": "Kathy! Kathy!",
      "emotions": [
        { "label": "fear", "confidence": 0.21 },
        { "label": "surprise", "confidence": 0.18 },
        { "label": "neutral", "confidence": 0.12 }
      ],
      "sentiments": [
        { "label": "negative", "confidence": 0.51 },
        { "label": "neutral", "confidence": 0.30 },
        { "label": "positive", "confidence": 0.19 }
      ]
    }
  ]
}
```

### Example `curl` Request

```bash
curl -X POST -F "video=@clip.mp4" [http://127.0.0.1:8000/analyze/video](http://127.0.0.1:8000/analyze/video)
```

---

## Repository Directory Structure

```text
MoodLens/
├── backend/                                      # FastAPI + PyTorch backend service
│   ├── main.py                                   # API server entry point & CLI flags
│   ├── pyproject.toml                            # Python project metadata & dependencies
│   ├── uv.lock                                   # Deterministic uv dependency lockfile
│   ├── README.md                                 # Backend-specific training & CLI guide
│   ├── src/
│   │   ├── backend/                              # FastAPI routing, rate limiting & middleware
│   │   ├── deployment/
│   │   │   ├── app.py                            # Deployment application setup
│   │   │   ├── inference.py                      # Whisper ASR + multimodal utterance inference
│   │   │   ├── models.py                         # Deployment model architecture definitions
│   │   │   ├── video_chunker.py                  # 60s FFmpeg video splitter & timestamp merger
│   │   │   └── analytics_aggregator.py           # Valence trajectory, mood shifts & CSV/JSON export
│   │   ├── streamlit/                            # Legacy Streamlit prototype UI
│   │   └── training/
│   │       ├── models.py                         # Encoders, FusionLayer, class weights & trainer
│   │       ├── meld_dataset.py                   # MELD CSV + MP4 + Mel-Spectrogram DataLoader
│   │       ├── train.py                          # Training entry point with TensorBoard logging
│   │       ├── count_parameters.py               # Trainable vs. frozen parameter counter
│   │       ├── install_ffmpeg.py                 # FFmpeg environment helper
│   │       ├── push_model.py                     # Hugging Face Hub model upload utility
│   │       ├── evaluate_metrics.py               # Per-class Precision/Recall/F1 & confusion matrices
│   │       ├── ablation_study.py                 # Unimodal & bimodal zero-masking ablation benchmark
│   │       ├── dataset_profiler.py               # MELD split distribution & ffprobe clip auditor
│   │       └── error_analysis.py                 # Modality-conflict rescue & high-confidence error audit
│   └── tests/
│       ├── __init__.py
│       ├── test_pipeline.py                      # Model forward-pass tensor shape & NaN checks
│       └── test_analytics_and_chunker.py         # Offline unit tests for analytics, profiler & chunker
├── frontend/                                     # Next.js + shadcn/ui web client
│   ├── app/
│   │   ├── layout.tsx                            # Root layout & navigation header
│   │   └── page.tsx                              # Video upload & real-time results view
│   └── components/ui/                            # Reusable shadcn/ui primitives
├── .gitignore
└── README.md
```

---

## Prerequisites & Local Development Setup

### Prerequisites

* **Python 3.12+** with [`uv`](https://github.com/astral-sh/uv) installed
* **FFmpeg** and **FFprobe** available on your system `PATH`:
  * **Windows:** `winget install Gyan.FFmpeg`
  * **macOS:** `brew install ffmpeg`
  * **Linux:** `sudo apt install ffmpeg`
* **Node.js 18+** and **pnpm** (for the Next.js frontend)

### 1. Start the Backend Server (Terminal 1)

```bash
cd backend
uv sync

# Start the API (default: http://127.0.0.1:8000)
$env:MOODLENS_ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
uv run python main.py
```

* Default API URL: `http://127.0.0.1:8000`
* Interactive Swagger UI: `http://127.0.0.1:8000/docs`
* Custom port / hot-reload: `uv run python main.py --port 9000 --reload`

### 2. Enable CORS for Browser Requests

CORS is disabled by default on the FastAPI backend. To allow requests from `http://localhost:3000`, set `MOODLENS_ALLOWED_ORIGINS` before launching `main.py`:

```bash
# Windows (PowerShell)
$env:MOODLENS_ALLOWED_ORIGINS="http://localhost:3000"
uv run python main.py

# Windows (Command Prompt)
set MOODLENS_ALLOWED_ORIGINS=http://localhost:3000
uv run python main.py

# macOS / Linux
MOODLENS_ALLOWED_ORIGINS="http://localhost:3000" uv run python main.py
```

### 3. Start the Frontend Client (Terminal 2)

```bash
cd frontend
pnpm install
pnpm dev
```

Open `http://localhost:3000`, select a video clip (`MP4`, `MOV`, `WebM`, or `AVI`), and click **Analyze video**.

---

## Environment Variables Reference

### Backend (`backend/`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MOODLENS_ALLOWED_ORIGINS` | `""` (empty) | Comma-separated browser origins permitted by CORS middleware |
| `MOODLENS_MAX_UPLOAD_MB` | `200` | Maximum accepted upload file size in megabytes |
| `MOODLENS_MAX_DURATION_SEC` | `600` | Maximum allowed video duration in seconds (10 minutes) |
| `MOODLENS_MAX_CONCURRENCY` | `1` | Maximum concurrent inference jobs allowed per worker |
| `MOODLENS_INFERENCE_TIMEOUT_SEC` | `900` | Hard timeout in seconds for a single video inference request |
| `MOODLENS_RATE_LIMIT_REQUESTS` | `10` | Maximum API requests permitted per rate-limit window |
| `MOODLENS_RATE_LIMIT_WINDOW_SEC` | `60` | Sliding window duration in seconds for rate limiting |
| `MOODLENS_MODEL_REPO` | `zeoxd/moodlens` | Hugging Face repository ID hosting trained `.pth` model weights |

### Frontend (`frontend/`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI inference backend |

---

## Evaluation, Profiling, Ablation & Error Analysis Suite

All analysis scripts inside `backend/src/training/` load saved checkpoints without altering model weights and export structured JSON reports.

### 1. MELD Dataset & Clip Integrity Profiler (`dataset_profiler.py`)

Audits CSV splits for missing/corrupted `.mp4` files, computes utterance word-length and clip-duration percentiles (`p50`, `p95`), cross-tabulates emotion vs. sentiment labels, and derives normalized inverse class weights:

```bash
cd backend
uv run python src/training/dataset_profiler.py \
  --csv path/to/train_sent_emo.csv \
  --video-dir path/to/train_splits \
  --probe-videos \
  --output dataset_profile.json
```

### 2. Per-Class F1 & Confusion Matrix Evaluator (`evaluate_metrics.py`)

Computes accuracy, Macro-F1, Weighted-F1, per-class Precision/Recall/F1, and full confusion matrices for both emotion (`7x7`) and sentiment (`3x3`) heads:

```bash
cd backend
uv run python src/training/evaluate_metrics.py \
  --checkpoint src/deployment/model/model.pth \
  --csv path/to/test_sent_emo.csv \
  --video-dir path/to/test_splits \
  --batch-size 8 \
  --output eval_report.json
```

### 3. Modality Ablation Benchmark (`ablation_study.py`)

Evaluates the trained checkpoint across 7 configurations (`full_multimodal`, `text_only`, `video_only`, `audio_only`, `text_video`, `text_audio`, `video_audio`) by zero-masking inactive encoder embeddings prior to `fusion_layer`:

```bash
cd backend
uv run python src/training/ablation_study.py \
  --checkpoint src/deployment/model/model.pth \
  --csv path/to/test_sent_emo.csv \
  --video-dir path/to/test_splits \
  --output ablation_report.json
```

### 4. Modality-Conflict & Error Analyzer (`error_analysis.py`)

Identifies utterances where multimodal fusion corrects text-only misclassifications (e.g., vocal tone or facial expression resolving sarcasm), measures cross-head polarity consistency, and ranks highest-confidence misclassifications:

```bash
cd backend
uv run python src/training/error_analysis.py \
  --checkpoint src/deployment/model/model.pth \
  --csv path/to/test_sent_emo.csv \
  --video-dir path/to/test_splits \
  --top-k 25 \
  --output error_analysis.json
```

---

## Automated Unit & Integrity Testing

### 1. Run Offline Analytics, Profiler & Chunker Tests

Runs in `< 1s` without downloading pretrained weights:

```bash
# From repository root (Windows PowerShell)
.\backend\.venv\Scripts\python.exe -m unittest discover -s backend/tests -t backend/tests -p "test_analytics_and_chunker.py" -v
```

### 2. Run Full Model Forward-Pass & Shape Tests

Instantiates `AudioEncoder` and `MultimodalSentimentModel`, verifies `(B, 7)` and `(B, 3)` output tensor dimensions, checks for `NaN` logits, and validates chunk offset aggregation:

```bash
# From repository root (Windows PowerShell)
.\backend\.venv\Scripts\python.exe -m unittest discover -s backend/tests -t backend/tests -v
```

---

## Troubleshooting

* **Browser `Network Error` or CORS block when clicking Analyze Video:** Ensure `MOODLENS_ALLOWED_ORIGINS="http://localhost:3000"` is exported in the same terminal session before starting `python main.py`.
* **`ffprobe` or `ffmpeg` not found:** Verify both binaries are installed and available on your system `PATH` (`ffmpeg -version` and `ffprobe -version`).
* **First-run model download delay:** On initial startup, `TextEncoder` (`bert-base-uncased`), `VideoEncoder` (`r3d_18`), Whisper, and the trained checkpoint from `MOODLENS_MODEL_REPO` are downloaded and cached locally.