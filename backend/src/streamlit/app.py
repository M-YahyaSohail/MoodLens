"""MoodLens Streamlit frontend.

Uploads a video to the MoodLens FastAPI endpoint and displays per-utterance
emotion & sentiment predictions in a minimal, friendly UI.

Run (while the API is up via `uv run python main.py`):
    uv run streamlit run src/streamlit/app.py
Launching the file directly with plain python also works:
    uv run python src/streamlit/app.py
"""

import hashlib
import logging
import os
import sys

import requests
import streamlit as st

# `streamlit` logs a scary "missing ScriptRunContext" warning for this check when
# the script runs outside its runtime; silence it.
logging.getLogger(
    "streamlit.runtime.scriptrunner_utils.script_run_context"
).setLevel(logging.ERROR)


def _ensure_streamlit_runtime():
    """Streamlit scripts must run under the Streamlit runtime (`streamlit run`).

    If this file was launched directly (e.g. `uv run python src/streamlit/app.py`),
    transparently hand it over to the Streamlit CLI so the page opens in a browser
    instead of crashing or warning in "bare mode".
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        return  # already running under `streamlit run` - no-op

    import streamlit.web.cli as stcli

    # Skip the one-time telemetry email prompt (it blocks non-interactive runs)
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    print(
        "Starting via the Streamlit runtime… a browser tab will open automatically.\n"
        "(use `streamlit run ...` directly for custom port/options)"
    )
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())


_ensure_streamlit_runtime()

API_URL_DEFAULT = os.environ.get("MOODLENS_API_URL", "http://127.0.0.1:8000")
UPLOAD_LIMIT_MB = int(os.environ.get("MOODLENS_MAX_UPLOAD_MB", "200"))
SUPPORTED_EXTENSIONS = ["mp4", "mov", "avi", "mkv", "webm", "mpg", "ts"]

EMOTION_COLORS = {
    "anger": "#d9534f",
    "disgust": "#9467bd",
    "fear": "#6a51a3",
    "joy": "#e8a020",
    "neutral": "#7f8fa6",
    "sadness": "#4a90d9",
    "surprise": "#e84393",
}

SENTIMENT_COLORS = {
    "negative": "#d9534f",
    "neutral": "#7f8fa6",
    "positive": "#2ecc71",
}

FRIENDLY_ERRORS = {
    400: "This file doesn't look like a supported video format "
         "(MP4, MOV, AVI, MKV, WebM…).",
    413: "The video is too large for the server's upload limit "
         f"({UPLOAD_LIMIT_MB} MB). Try a shorter or more compressed clip.",
    422: "The video is longer than the server allows. Try a shorter clip.",
    429: "Too many requests — the server's rate limit was hit. "
         "Wait a minute and try again.",
    503: "The server is busy analyzing another video right now. "
         "Try again in a moment.",
    504: "Analysis timed out. Try a shorter video.",
}

CSS = """
<style>
    .pill {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-weight: 600; font-size: 0.8rem; color: #ffffff; margin-right: 8px;
    }
    .row { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
    .lab { width: 96px; font-size: 0.82rem; font-weight: 600; flex-shrink: 0; }
    .track {
        flex: 1; height: 8px; background: #ececf3; border-radius: 6px;
        overflow: hidden;
    }
    .fill { height: 100%; border-radius: 6px; }
    .pct { font-size: 0.78rem; color: #666; width: 40px; text-align: right; }
    .quote {
        font-style: italic; color: #444; margin: 4px 0 10px 0;
        line-height: 1.4;
    }
</style>
"""


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def confidence_bar(label: str, confidence: float, color: str) -> str:
    pct = max(0.0, min(1.0, confidence)) * 100
    return (
        f'<div class="row">'
        f'<span class="lab" style="color:{color}">{label}</span>'
        f'<div class="track"><div class="fill" style="width:{pct:.0f}%;'
        f'background:{color}"></div></div>'
        f'<span class="pct">{confidence:.2f}</span>'
        f"</div>"
    )


def call_analyze_api(api_url: str, video_bytes: bytes, filename: str) -> dict:
    """POST the video to the API and return (ok, payload) tuple."""
    response = requests.post(
        f"{api_url.rstrip('/')}/analyze/video",
        files={"video": (filename, video_bytes, "video/mp4")},
        timeout=(5, 900),
    )
    if response.status_code == 200:
        return True, response.json()

    detail = FRIENDLY_ERRORS.get(response.status_code)
    reason = detail
    if detail is None:
        try:
            reason = response.json().get("detail", response.reason)
        except Exception:
            reason = response.reason
        reason = f"Unexpected API error ({response.status_code}): {reason}"
    else:
        reason = response.json().get("detail") or detail
    return False, reason


def render_utterance(utterance: dict) -> None:
    with st.container(border=True):
        top_emotion = utterance["emotions"][0]
        top_sentiment = utterance["sentiments"][0]

        col1, col2, col3 = st.columns([1, 2, 2])
        col1.markdown(
            f'<span class="pill" style="background:'
            f'{EMOTION_COLORS[top_emotion["label"]]};">'
            f'{top_emotion["label"]}</span>'
            f'<span class="pill" style="background:'
            f'{SENTIMENT_COLORS[top_sentiment["label"]]};">'
            f'{top_sentiment["label"]}</span>',
            unsafe_allow_html=True,
        )
        col2.caption(
            f"{fmt_time(utterance['start_time'])} → {fmt_time(utterance['end_time'])}"
        )
        col3.caption(f"top confidence {top_emotion['confidence']:.0%}")

        if utterance.get("text"):
            st.markdown(f'<div class="quote">“{utterance["text"]}”</div>', unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.caption("Emotions")
            for item in utterance["emotions"]:
                color = EMOTION_COLORS.get(item["label"], "#888888")
                st.markdown(
                    confidence_bar(item["label"], item["confidence"], color),
                    unsafe_allow_html=True,
                )
        with right:
            st.caption("Sentiments")
            for item in utterance["sentiments"]:
                color = SENTIMENT_COLORS.get(item["label"], "#888888")
                st.markdown(
                    confidence_bar(item["label"], item["confidence"], color),
                    unsafe_allow_html=True,
                )


def render_results(utterances: list) -> None:
    if not utterances:
        st.info(
            "No speech segments were detected in this clip, so there is "
            "nothing to analyze. Try a video with audible dialogue."
        )
        return

    n_utterances = len(utterances)
    st.success(f"Analyzed **{n_utterances}** speech segment"
               f"{'s' if n_utterances != 1 else ''}.")

    for utterance in utterances:
        render_utterance(utterance)


def check_health(api_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        if response.status_code == 200:
            state = "model ready" if response.json().get("model_loaded") else "loading model"
            return True, state
        return False, f"unhealthy (HTTP {response.status_code})"
    except requests.ConnectionError:
        return False, "unreachable"


# --- Page setup ---------------------------------------------------------------

st.set_page_config(page_title="MoodLens", page_icon="🎬", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

st.title("MoodLens 🎬")
st.caption("Multimodal emotion & sentiment analysis — upload a video clip and "
           "every spoken utterance gets classified by a local deep model.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API address", value=API_URL_DEFAULT)
    healthy, state = check_health(api_url)
    st.caption(f"Server status: {'🟢' if healthy else '🔴'} {state}")

    st.divider()
    st.caption(
        f"Limits: uploads ≤ **{UPLOAD_LIMIT_MB} MB**, max video length "
        f"**10 min**, one analysis at a time."
    )
    st.caption("Start the API with: `uv run python main.py`")

# --- Upload & analyze ----------------------------------------------------------

uploaded = st.file_uploader(
    "Choose a video (MP4, MOV, AVI, MKV, WebM)",
    type=SUPPORTED_EXTENSIONS,
)

if uploaded is None:
    st.info("⬆️ Upload a clip to get started.")
    st.stop()

size_mb = uploaded.size / (1024 * 1024)
st.video(uploaded.getvalue())

if size_mb > UPLOAD_LIMIT_MB:
    st.error(
        f"Your file is {size_mb:.0f} MB but the server accepts at most "
        f"{UPLOAD_LIMIT_MB} MB (MOODLENS_MAX_UPLOAD_MB). "
        "Trim or compress the clip and retry."
    )
    st.stop()

# Cached per uploaded file so unrelated interactions don't wipe results
file_key = hashlib.sha1(uploaded.getvalue()).hexdigest()
st.session_state.setdefault("results", {})
st.session_state.setdefault("error", {})

analyze = st.button("Analyze video", type="primary", use_container_width=True)

if analyze:
    with st.spinner(
        "Transcribing speech and running multimodal inference… "
        "this can take a while for longer videos."
    ):
        try:
            ok, payload = call_analyze_api(api_url, uploaded.getvalue(), uploaded.name)
        except requests.ConnectionError:
            ok, payload = (
                False,
                f"Could not reach the API at `{api_url}`. Make sure it is "
                "running (`uv run python main.py`) and the address matches.",
            )
        except requests.Timeout:
            ok, payload = (
                False,
                "The request took too long and was aborted. Try a shorter video.",
            )
        if ok:
            st.session_state["results"][file_key] = payload
            st.session_state["error"].pop(file_key, None)
        else:
            st.session_state["error"][file_key] = payload
            st.session_state["results"].pop(file_key, None)

if file_key in st.session_state["error"]:
    st.error(st.session_state["error"][file_key])

if file_key in st.session_state["results"]:
    render_results(st.session_state["results"][file_key].get("utterances", []))
