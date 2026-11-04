import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave

import cv2
import numpy as np
import torch
import torchaudio
import whisper
from src.deployment.models import MultimodalSentimentModel
from transformers import AutoTokenizer

EMOTION_MAP = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise",
}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def _load_pcm_wav(audio_path):
    """Load a PCM WAV produced by ffmpeg without torchaudio's I/O backend.

    torchaudio.load requires the torchcodec extension in torch>=2.15, so we
    decode the ffmpeg-generated 16 kHz mono WAV directly with the stdlib.
    """
    with wave.open(audio_path, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    waveform = torch.from_numpy(samples.astype(np.float32) / 32768.0).unsqueeze(0)
    return waveform, sample_rate

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_REPO_ID = os.environ.get("MOODLENS_MODEL_REPO", "zeoxd/moodlens")


def ensure_ffmpeg():
    """Verify that the FFmpeg binary is available on PATH for local use."""
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        print("FFmpeg not found on PATH.")
        print("Install it so the 'ffmpeg' command is accessible, e.g.:")
        print(
            "  - Windows: winget install Gyan.FFmpeg  (or download from https://ffmpeg.org)"
        )
        print("  - macOS:   brew install ffmpeg")
        print("  - Linux:   sudo apt install ffmpeg")
        return False

    print(f"FFmpeg found: {ffmpeg_path}")
    return True


class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


class AudioProcessor:
    def extract_features(self, video_path, max_length=300):
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    audio_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            waveform, sample_rate = _load_pcm_wav(audio_path)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(
        self, video_path, start_time, end_time, temp_dir=tempfile.gettempdir()
    ):
        # Reject empty/negative-length segments (Whisper can emit start==end
        # markers); ffmpeg fails on these and floods logs otherwise
        if end_time <= start_time:
            raise ValueError(f"Segment has invalid duration: {start_time} - {end_time}")

        os.makedirs(temp_dir, exist_ok=True)
        # Unique name prevents collisions between concurrent processes/users
        segment_path = os.path.join(
            temp_dir, f"segment_{uuid.uuid4().hex}_{start_time}_{end_time}.mp4"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-y",
                segment_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction failed: " + segment_path)

        return segment_path


def _resolve_model_path():
    """Locate the trained weights, fetching them from the Hugging Face Hub.

    The weights live in the `zeoxd/moodlens` repo on the Hub and are downloaded
    with huggingface_hub, which caches them locally so the transfer happens
    only once. If the Hub is unreachable, an already-present local file is
    used as a fallback (useful for offline machines / air-gapped setups).
    """

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="model.pth")
        print(f"Model weights resolved from Hugging Face repo '{MODEL_REPO_ID}'")
        return path
    except Exception as e:
        print(
            f"Hugging Face download failed ({str(e)}). "
            "Falling back to a local model.pth if available."
        )

    local_candidates = [
        os.path.join(DEFAULT_MODEL_DIR, "model.pth"),
        os.path.join("model", "model.pth"),
    ]
    for path in local_candidates:
        if os.path.exists(path):
            print(f"Using local model weights: {path}")
            return path

    raise FileNotFoundError(
        f"Unable to fetch model weights from '{MODEL_REPO_ID}' and no local "
        "model.pth was found."
    )


def model_fn():
    # Load the model for inference
    if not ensure_ffmpeg():
        raise RuntimeError("FFmpeg installation failed - required for inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)

    model_path = _resolve_model_path()

    print("Loading model from path: " + model_path)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    return {
        "model": model,
        "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "transcriber": whisper.load_model(
            "base",
            device="cpu" if device.type == "cpu" else device,
        ),
        "device": device,
    }


def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    video_path = input_data["video_path"]

    result = model_dict["transcriber"].transcribe(video_path, word_timestamps=True)

    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result["segments"]:
        segment_path = None
        try:
            segment_path = utterance_processor.extract_segment(
                video_path, segment["start"], segment["end"]
            )

            video_frames = utterance_processor.video_processor.process_video(
                segment_path
            )
            audio_features = utterance_processor.audio_processor.extract_features(
                segment_path
            )
            text_inputs = tokenizer(
                segment["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            # Move to device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            # Get predictions
            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

            predictions.append(
                {
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"],
                    "emotions": [
                        {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()}
                        for idx, conf in zip(emotion_indices, emotion_values)
                    ],
                    "sentiments": [
                        {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()}
                        for idx, conf in zip(sentiment_indices, sentiment_values)
                    ],
                }
            )

        except Exception as e:
            print("Segment failed inference: " + str(e))

        finally:
            # Cleanup
            if segment_path and os.path.exists(segment_path):
                os.remove(segment_path)
    return {"utterances": predictions}


def process_local_video(video_path):
    model_dict = model_fn()

    input_data = {"video_path": video_path}

    predictions = predict_fn(input_data, model_dict)

    for utterance in predictions["utterances"]:
        print("\nUtterance:")
        print(f"""Start: {utterance['start_time']}s, End: {
              utterance['end_time']}s""")
        print(f"Text: {utterance['text']}")
        print("\n Top Emotions:")
        for emotion in utterance["emotions"]:
            print(f"{emotion['label']}: {emotion['confidence']:.2f}")
        print("\n Top Sentiments:")
        for sentiment in utterance["sentiments"]:
            print(f"{sentiment['label']}: {sentiment['confidence']:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.deployment.inference <path-to-video.mp4>")
        sys.exit(1)

    process_local_video(sys.argv[1])
