# MoodLens

Multimodal Emotion & Sentiment Analysis on the MELD dataset.

MoodLens transcribes the speech in a video, then runs each detected speech
segment through a PyTorch multimodal model (text + video + audio) trained on
MELD to predict per-segment **emotion** and **sentiment** probabilities.

## Architecture

| Part | Technology | Responsibility |
|------|-----------|----------------|
| Backend | FastAPI + PyTorch (`backend/`) | Model serving, Whisper transcription, multimodal inference |
| Frontend | Next.js + shadcn/ui (`frontend/`) | Video upload, results display (emojis + progress bars) |

### API contract

The FastAPI backend exposes:

- `GET /health` — liveness + model-loaded status
- `POST /analyze/video` — multipart upload (form field `video`) → returns per-segment predictions

Example response:

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

Emotions: `anger, disgust, fear, joy, neutral, sadness, surprise`
Sentiments: `negative, neutral, positive`

---

## Prerequisites

- **Python** with [`uv`](https://github.com/astral-sh/uv) (backend)
- **FFmpeg** on your `PATH` (required by the backend for audio/video processing)
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **Node.js** + **pnpm** (frontend)

---

## Running the full application

You need two terminals: one for the FastAPI backend and one for the Next.js frontend.

### 1. Backend (FastAPI)

From the repository root:

```bash
# Install Python dependencies
cd backend
uv sync

# Start the API (default: http://127.0.0.1:8000)
uv run python main.py
```

Options: `uv run python main.py --port 9000 --reload`

Interactive API docs: `http://127.0.0.1:8000/docs`

### 2. Frontend (Next.js)

From the repository root, in a **separate** terminal:

```bash
cd frontend

# Install dependencies (first time only)
pnpm install

# Run the dev server (default: http://localhost:3000)
pnpm dev
```

Open `http://localhost:3000` in your browser, upload a video file, and click
**Analyze video**. The frontend sends the file to the FastAPI backend and
displays per-segment emotion/sentiment probabilities using emojis and progress
bars.

### 3. Enable CORS (required for the browser frontend)

The FastAPI backend disables CORS by default. The Next.js frontend calls the
API from `http://localhost:3000`, so you must allow that origin. Run the
backend with:

```bash
# Windows (PowerShell)
$env:MOODLENS_ALLOWED_ORIGINS="http://localhost:3000"
uv run python main.py

# Windows (cmd)
set MOODLENS_ALLOWED_ORIGINS=http://localhost:3000
uv run python main.py

# macOS / Linux
MOODLENS_ALLOWED_ORIGINS="http://localhost:3000" uv run python main.py
```

If you host the frontend elsewhere, add that origin instead (comma-separated
for multiple origins).

---

## Environment variables

### Backend (`backend/`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOODLENS_ALLOWED_ORIGINS` | `` (empty) | Comma-separated list of browser origins allowed via CORS |
| `MOODLENS_MAX_UPLOAD_MB` | `200` | Max upload size in MB |
| `MOODLENS_MAX_DURATION_SEC` | `600` | Max video duration in seconds |
| `MOODLENS_MAX_CONCURRENCY` | `1` | Max concurrent inference jobs |
| `MOODLENS_INFERENCE_TIMEOUT_SEC` | `900` | Inference timeout in seconds |
| `MOODLENS_RATE_LIMIT_REQUESTS` | `10` | Requests per rate-limit window |
| `MOODLENS_RATE_LIMIT_WINDOW_SEC` | `60` | Rate-limit window in seconds |
| `MOODLENS_MODEL_REPO` | `zeoxd/moodlens` | Hugging Face repo for model weights |

### Frontend (`frontend/`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI backend |

---

## Project layout

```
MoodLens/
├── backend/              # FastAPI + PyTorch service
│   ├── main.py           # API entry point
│   ├── pyproject.toml    # Python dependencies
│   └── src/
│       ├── deployment/   # FastAPI app, inference, models
│       ├── streamlit/    # Streamlit UI (legacy)
│       └── training/     # Training code
├── frontend/             # Next.js + shadcn/ui web app
│   ├── app/
│   │   ├── layout.tsx    # Root layout with site header
│   │   └── page.tsx      # Upload + results page (client component)
│   └── components/ui/    # shadcn/ui components (button, card, progress, ...)
└── README.md
```

---

## Testing end-to-end

1. Start the backend (step 1) — wait for `Model ready - accepting requests`.
2. Start the frontend (step 2) — open `http://localhost:3000`.
3. Upload a video clip (MP4/MOV/WebM/AVI, ≤ 200 MB, ≤ 10 min).
4. Click **Analyze video**.
5. View per-segment emotion & sentiment probabilities with emojis and progress bars.

A sample `curl` request against the backend:

```bash
curl -X POST -F "video=@clip.mp4" http://127.0.0.1:8000/analyze/video
```

## Backend details

See `backend/README.md` for training, local CLI inference, Streamlit UI, and
full API safety-limit documentation.
