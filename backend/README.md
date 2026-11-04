# MoodLens Backend

Multimodal emotion & sentiment analysis (text + video + audio) on the MELD dataset,
built with PyTorch. Fully local — no cloud services required.

## Setup

```bash
uv sync        # install dependencies
```

FFmpeg must be available on your PATH (checked automatically at startup):

```bash
winget install Gyan.FFmpeg   # Windows
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Linux
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/training/models.py` | Model architecture + trainer |
| `src/training/meld_dataset.py` | MELD dataset & dataloaders |
| `src/training/train.py` | Training entry point |
| `src/deployment/models.py` | Inference-time model architecture |
| `src/deployment/inference.py` | Inference pipeline (Whisper + model) |
| `src/deployment/model/model.pth` | Trained weights |
| `src/deployment/app.py` | FastAPI service |
| `src/streamlit/app.py` | Streamlit web UI |
| `dataset/` | MELD dataset (`train`, `dev`, `test`) |

## Training

Run from the `backend/` directory:

```bash
uv run python -m src.training.train --epochs 20 --batch-size 16
```

Best checkpoint is saved as `model.pth` in the `--model-dir` (project root by
default). TensorBoard logs go to `runs/`. The data dirs default to the local
`dataset/` folder and can be overridden with `--train-dir`, `--val-dir`,
`--test-dir`.

## Local CLI inference

```bash
uv run python -m src.deployment.inference path/to/video.mp4
```

## Web UI (Streamlit)

With the API running (`uv run python main.py`), start the frontend:

```bash
uv run streamlit run src/streamlit/app.py
# or simply:
uv run python src/streamlit/app.py   # auto-hands off to the Streamlit runtime
```

A browser tab opens at `http://localhost:8501`. For custom options (port,
autoreload), use the `streamlit run` form directly.

A browser tab opens at `http://localhost:8501`. Upload a video, click
**Analyze video**, and every detected speech segment is shown with its text,
timestamps, and top-3 emotion/sentiment confidence bars. The API address is
configurable in the sidebar (`MOODLENS_API_URL` or the text input), which also
shows live server health.

## REST API

```bash
uv run python main.py
# or: uv run uvicorn src.deployment.app:app --host 127.0.0.1 --port 8000
```

Options: `--port 9000` to change the port, `--host 0.0.0.0` to expose the API
to your local network, `--reload` for development auto-restarts.

Interactively explore the endpoints via the auto docs at
`http://127.0.0.1:8000/docs`.

### Endpoints

- `GET /health` — liveness + model-loaded status.
- `POST /analyze/video` — multipart upload of a video file. Whisper transcribes
  the speech locally, each utterance segment is analyzed by the multimodal
  model, and per-segment predictions are returned:

```json
{
  "utterances": [
    {
      "start_time": 0.0,
      "end_time": 1.6,
      "text": "Kathy! Kathy!",
      "emotions":   [ {"label": "fear",     "confidence": 0.21}, ... ],
      "sentiments": [ {"label": "negative", "confidence": 0.51}, ... ]
    }
  ]
}
```

Example:

```bash
curl -X POST -F "video=@some_clip.mp4" http://127.0.0.1:8000/analyze/video
```

### Safety limits

All limits are overridable via environment variables. Defaults are tuned to
prevent overloading the local pipeline.

| Guard | Behavior | Env var (default) |
| --- | --- | --- |
| Upload size cap | `413` when exceeded (enforced chunk-by-chunk while streaming) | `MOODLENS_MAX_UPLOAD_MB` (200) |
| Video duration cap | `422` for longer videos | `MOODLENS_MAX_DURATION_SEC` (600) |
| Concurrency limit | `503` while another analysis is running | `MOODLENS_MAX_CONCURRENCY` (1) |
| Inference timeout | `504` if analysis exceeds the budget | `MOODLENS_INFERENCE_TIMEOUT_SEC` (900) |
| Rate limit | `429` + `Retry-After` per client IP (sliding window) | `MOODLENS_RATE_LIMIT_REQUESTS` (10) / `MOODLENS_RATE_LIMIT_WINDOW_SEC` (60) |
| Type validation | `400` — validates actual binary header (magic bytes), not the client-supplied filename/content type | — |

Other protections:

- Client filenames are never used; uploads get random temp names and are
  always deleted after processing.
- Error responses are generic; stack traces only appear in server logs.
- CORS is disabled by default; enable for specific browser origins with
  `MOODLENS_ALLOWED_ORIGINS="http://localhost:5173,https://myapp.example"`.
