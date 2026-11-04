"""FastAPI application exposing MoodLens multimodal analysis over HTTP.

Endpoint
--------
POST /analyze/video — multipart upload of a video file, runs the local
inference pipeline (whisper transcription + multimodal model) and returns
per-utterance emotion/sentiment predictions.

Security / anti-abuse measures
------------------------------
- Upload never trusted by extension or client-supplied content type; the
  actual binary header (magic bytes) is validated instead.
- Streaming upload with a hard size cap enforced chunk-by-chunk (uploads are
  never fully buffered in memory).
- Video duration cap (rejects crafted/huge-duration files before inference).
- Per-client sliding-window rate limiting (429 + Retry-After).
- Inference concurrency limit (503 when one job is already running).
- Hard timeout on inference (504) so hung jobs cannot stall the service.
- Client filenames are never used; temp files get random names in the system
  temporary directory and are always deleted.
- Error responses are generic; internal details only go to logs.
- Permissive CORS is disabled by default (opt-in via MOODLENS_ALLOWED_ORIGINS).

Run:  uv run uvicorn src.deployment.app:app --host 127.0.0.1 --port 8000
"""

import asyncio
import logging
import os
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.deployment.inference import ensure_ffmpeg, model_fn, predict_fn

# --- Configuration (all values overridable via environment variables) ---
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MOODLENS_MAX_UPLOAD_MB", "200"))
MAX_VIDEO_DURATION_SEC = int(os.environ.get("MOODLENS_MAX_DURATION_SEC", "600"))
MAX_CONCURRENT_INFERENCE = int(os.environ.get("MOODLENS_MAX_CONCURRENCY", "1"))
INFERENCE_TIMEOUT_SEC = int(os.environ.get("MOODLENS_INFERENCE_TIMEOUT_SEC", "900"))
RATE_LIMIT_REQUESTS = int(os.environ.get("MOODLENS_RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("MOODLENS_RATE_LIMIT_WINDOW_SEC", "60"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("MOODLENS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
CHUNK_SIZE = 1024 * 1024  # 1 MiB

logger = logging.getLogger("moodlens.api")

# Single long-running state, initialized on startup
_MODEL: dict | None = None

# Inference runs on a bounded thread pool so it never degrades API responsiveness
_inference_slots = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
_inference_executor = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_INFERENCE, thread_name_prefix="moodlens"
)


# --- Validation helpers -----------------------------------------------------


class UploadValidationError(Exception):
    """Uploaded data failed a safety/local-integrity validation check."""


class VideoTooLargeError(UploadValidationError):
    """Upload exceeded the configured size cap."""


class VideoTooLongError(UploadValidationError):
    """Video exceeded the configured maximum duration."""


class VideoNotReadableError(UploadValidationError):
    """Video could not be opened or probed (likely corrupt/fake)."""


def _is_allowed_video_header(head: bytes) -> bool:
    """Validate the file's binary header instead of trusting client metadata."""
    if len(head) >= 12 and head[4:8] == b"ftyp":  # MP4 / MOV (isom families)
        return True
    if head[:4] == b"RIFF":  # AVI
        return True
    if head[:4] == b"\x1a\x45\xdf\xa3":  # MKV / WebM
        return True
    if head[:4] == b"\x00\x00\x01\xba":  # MPEG-PS (VOB/MPG)
        return True
    if len(head) >= 189 and head[0] == 0x47 and head[188] == 0x47:  # MPEG-TS
        return True
    return False


async def save_upload_to_temp_file(file: UploadFile) -> str:
    """Stream the upload to a randomly-named temp file with a hard size cap.

    The client-provided filename is ignored entirely (no path traversal or
    name-based tricks). Returns the temp path; raises on invalid input.
    """
    fd, temp_path = tempfile.mkstemp(prefix="moodlens_", suffix=".mp4")

    first_chunk = await file.read(CHUNK_SIZE)

    if not first_chunk:
        os.close(fd)
        os.remove(temp_path)
        raise UploadValidationError("Empty upload: no file content received")

    if not _is_allowed_video_header(first_chunk):
        os.close(fd)
        os.remove(temp_path)
        raise UploadValidationError(
            "Unsupported file type: content is not a recognized video format"
        )

    if len(first_chunk) > MAX_UPLOAD_BYTES:
        os.close(fd)
        os.remove(temp_path)
        raise VideoTooLargeError(
            f"Uploaded file exceeds the {MAX_UPLOAD_SIZE_MB} MB limit"
        )

    total = len(first_chunk)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(first_chunk)
            while chunk := await file.read(CHUNK_SIZE):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise VideoTooLargeError(
                        f"Uploaded file exceeds the {MAX_UPLOAD_SIZE_MB} MB limit"
                    )
                out.write(chunk)
    except BaseException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise

    logger.info("Received upload of %.1f MB", total / (1024 * 1024))
    return temp_path


def _probe_duration(path: str) -> float:
    """Read the video's fps/frame metadata to get its duration in seconds."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoNotReadableError("Video could not be opened for inspection")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()

    if not fps or fps <= 0 or not frame_count or frame_count <= 0:
        raise VideoNotReadableError("Video metadata unreadable (file may be corrupt)")

    return frame_count / fps


class SlidingWindowRateLimiter:
    """Simple in-memory per-client rate limiter (no external dependencies)."""

    def __init__(self, limit: int, window_sec: int):
        self._limit = limit
        self._window_sec = window_sec
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window_sec
        hits = self._hits[client_key]

        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self._limit:
            return False

        hits.append(now)

        # Opportunistic cleanup so idle clients don't grow memory unbounded
        if len(self._hits) > 10_000:
            for key, recent in list(self._hits.items()):
                if not recent or recent[-1] <= cutoff:
                    del self._hits[key]

        return True


_rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SEC)


# --- Inference ---------------------------------------------------------------


async def _run_inference(video_path: str) -> dict:
    def _worker():
        return predict_fn({"video_path": video_path}, _MODEL)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_inference_executor, _worker)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MODEL

    if not ensure_ffmpeg():
        raise RuntimeError(
            "FFmpeg not found on PATH - required for inference. Install it "
            "(e.g. winget install Gyan.FFmpeg) and restart the API."
        )

    logger.info("Loading model, tokenizer and Whisper transcriber...")
    loop = asyncio.get_running_loop()
    _MODEL = await loop.run_in_executor(None, model_fn)
    logger.info("Model ready - accepting requests")

    yield

    _inference_executor.shutdown(wait=False)


app = FastAPI(
    title="MoodLens API",
    description="Multimodal emotion & sentiment analysis for videos (local inference)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS is disabled by default; only enable for explicitly allowed origins
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# --- Endpoints ----------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None,
    }


@app.post("/analyze/video")
async def analyze_video(request: Request, video: UploadFile = File(...)):
    # 1. Per-client rate limiting
    client_key = request.client.host if request.client else "unknown"
    if not _rate_limiter.allow(client_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before sending another video.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SEC)},
        )

    # 2. Cheap early size check before consuming any body bytes
    declared_length = request.headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > MAX_UPLOAD_BYTES + 2 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Request body exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit"
                    ),
                )
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Malformed Content-Length header"
            )

    # 3. Fail fast if all inference capacity is taken
    if _inference_slots.locked() or _MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server busy: the analysis pipeline is at capacity. Try again shortly."
            ),
            headers={"Retry-After": "30"},
        )

    temp_path = None
    try:
        # 4. Streaming validation + capped write to disk
        temp_path = await save_upload_to_temp_file(video)

        # 5. Duration cap (fake/crafted very-long-duration files are the typical
        #    DoS vector for automated transcription pipelines)
        duration = await asyncio.to_thread(_probe_duration, temp_path)
        if duration > MAX_VIDEO_DURATION_SEC:
            raise VideoTooLongError(
                f"Video is {duration:.0f}s; the limit is {MAX_VIDEO_DURATION_SEC}s"
            )
        logger.info("Analyzing video of %.1f seconds for client %s", duration, client_key)

        # 6. Inference with concurrency limit and hard timeout
        async with _inference_slots:
            result = await asyncio.wait_for(
                _run_inference(temp_path), timeout=INFERENCE_TIMEOUT_SEC
            )

        return JSONResponse(status_code=200, content=result)

    except VideoTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except VideoTooLongError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except UploadValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError:
        logger.error("Inference timed out after %ss", INFERENCE_TIMEOUT_SEC)
        raise HTTPException(
            status_code=504,
            detail="Analysis timed out. Try a shorter video.",
        )
    except HTTPException:
        raise
    except Exception:
        # Never leak library stack traces to clients - log them instead
        logger.exception("Unhandled error while processing upload")
        raise HTTPException(
            status_code=500, detail="Internal error while processing the video"
        )
    finally:
        # 7. Temporary file is always cleaned up, success or failure
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Failed to remove temporary file: %s", temp_path)
