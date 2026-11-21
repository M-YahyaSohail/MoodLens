from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from typing import Any


@dataclass(frozen=True)
class VideoChunk:
    chunk_index: int
    file_path: str
    start_offset: float
    duration: float


def get_video_duration(video_path: str | Path) -> float:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_video_into_chunks(
    video_path: str | Path,
    output_dir: str | Path,
    max_chunk_duration: float = 60.0,
) -> list[VideoChunk]:
    if max_chunk_duration <= 0:
        raise ValueError("max_chunk_duration must be greater than zero.")

    source = Path(video_path)
    total_duration = get_video_duration(source)

    if total_duration <= max_chunk_duration:
        return [
            VideoChunk(
                chunk_index=0,
                file_path=str(source),
                start_offset=0.0,
                duration=round(total_duration, 3),
            )
        ]

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = math.ceil(total_duration / max_chunk_duration)
    chunks: list[VideoChunk] = []

    for idx in range(num_chunks):
        start_time = idx * max_chunk_duration
        clip_duration = min(max_chunk_duration, total_duration - start_time)
        chunk_file = target_dir / f"{source.stem}_chunk_{idx:03d}{source.suffix}"

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-t",
            f"{clip_duration:.3f}",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            str(chunk_file),
        ]
        subprocess.run(command, capture_output=True, text=True, check=True)

        chunks.append(
            VideoChunk(
                chunk_index=idx,
                file_path=str(chunk_file),
                start_offset=round(start_time, 3),
                duration=round(clip_duration, 3),
            )
        )

    return chunks


def merge_chunk_predictions(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    merged_utterances: list[dict[str, Any]] = []
    emotion_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()

    for chunk in sorted(chunk_results, key=lambda item: float(item.get("start_offset", 0.0))):
        offset = float(chunk.get("start_offset", 0.0))
        for utterance in chunk.get("utterances", []):
            shifted = dict(utterance)
            if "start" in shifted:
                shifted["start"] = round(float(shifted["start"]) + offset, 3)
            if "end" in shifted:
                shifted["end"] = round(float(shifted["end"]) + offset, 3)

            emotion = shifted.get("emotion")
            sentiment = shifted.get("sentiment")
            if isinstance(emotion, str) and emotion:
                emotion_counts[emotion] += 1
            if isinstance(sentiment, str) and sentiment:
                sentiment_counts[sentiment] += 1

            merged_utterances.append(shifted)

    total = len(merged_utterances)
    overall_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"
    overall_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral"

    emotion_distribution = (
        {label: round(count / total, 4) for label, count in emotion_counts.items()}
        if total > 0
        else {}
    )
    sentiment_distribution = (
        {label: round(count / total, 4) for label, count in sentiment_counts.items()}
        if total > 0
        else {}
    )

    return {
        "utterances": merged_utterances,
        "overall_emotion": overall_emotion,
        "overall_sentiment": overall_sentiment,
        "emotion_distribution": emotion_distribution,
        "sentiment_distribution": sentiment_distribution,
    }