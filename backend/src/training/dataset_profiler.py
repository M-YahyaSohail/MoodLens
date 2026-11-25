import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EMOTION_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def compute_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)

    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(sorted_values[lower], 4)

    weight = rank - lower
    interpolated = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    return round(interpolated, 4)


def summarize_numeric_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    ordered = sorted(values)
    mean_val = sum(ordered) / len(ordered)
    return {
        "min": round(ordered[0], 4),
        "mean": round(mean_val, 4),
        "p50": compute_percentile(ordered, 50.0),
        "p95": compute_percentile(ordered, 95.0),
        "max": round(ordered[-1], 4),
    }


def probe_clip_duration(clip_path: Path) -> float | None:
    if not clip_path.is_file():
        return None

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(clip_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None

    try:
        duration = float(result.stdout.strip())
        return duration if duration > 0 else None
    except ValueError:
        return None


def compute_inverse_weights(counts: Counter[str], label_order: list[str]) -> dict[str, float]:
    raw_weights: dict[str, float] = {}
    for label in label_order:
        count = counts.get(label, 0)
        raw_weights[label] = (1.0 / count) if count > 0 else 0.0

    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        return dict.fromkeys(label_order, 0.0)

    return {label: round(weight / total_weight, 4) for label, weight in raw_weights.items()}


def profile_meld_split(
    csv_path: str | Path,
    video_dir: str | Path | None = None,
    probe_videos: bool = False,
    output_path: str | Path = "dataset_profile.json",
) -> dict[str, Any]:
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"MELD CSV not found: {csv_file}")

    clips_root = Path(video_dir) if video_dir else None

    total_rows = 0
    missing_clips: list[str] = []
    corrupted_clips: list[str] = []
    emotion_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    cross_tab: dict[str, Counter[str]] = defaultdict(Counter)
    speaker_counts: Counter[str] = Counter()
    word_lengths: list[float] = []
    clip_durations: list[float] = []

    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1

            utterance = (row.get("Utterance") or "").strip()
            emotion = (row.get("Emotion") or "").strip().lower()
            sentiment = (row.get("Sentiment") or "").strip().lower()
            speaker = (row.get("Speaker") or "Unknown").strip()
            dialogue_id = (row.get("Dialogue_ID") or "").strip()
            utterance_id = (row.get("Utterance_ID") or "").strip()

            if emotion:
                emotion_counts[emotion] += 1
            if sentiment:
                sentiment_counts[sentiment] += 1
            if emotion and sentiment:
                cross_tab[emotion][sentiment] += 1
            if speaker:
                speaker_counts[speaker] += 1

            tokens = [tok for tok in utterance.split() if tok]
            word_lengths.append(float(len(tokens)))

            if clips_root is not None and dialogue_id and utterance_id:
                clip_name = f"dia{dialogue_id}_utt{utterance_id}.mp4"
                clip_path = clips_root / clip_name
                if not clip_path.is_file():
                    missing_clips.append(clip_name)
                elif probe_videos:
                    duration = probe_clip_duration(clip_path)
                    if duration is None:
                        corrupted_clips.append(clip_name)
                    else:
                        clip_durations.append(duration)

    report: dict[str, Any] = {
        "csv_path": str(csv_file),
        "total_utterances": total_rows,
        "unique_speakers": len(speaker_counts),
        "top_speakers": dict(speaker_counts.most_common(10)),
        "emotion_distribution": {
            label: {
                "count": emotion_counts.get(label, 0),
                "ratio": round(emotion_counts.get(label, 0) / total_rows, 4)
                if total_rows > 0
                else 0.0,
                "normalized_class_weight": compute_inverse_weights(emotion_counts, EMOTION_ORDER)[
                    label
                ],
            }
            for label in EMOTION_ORDER
        },
        "sentiment_distribution": {
            label: {
                "count": sentiment_counts.get(label, 0),
                "ratio": round(sentiment_counts.get(label, 0) / total_rows, 4)
                if total_rows > 0
                else 0.0,
                "normalized_class_weight": compute_inverse_weights(
                    sentiment_counts, SENTIMENT_ORDER
                )[label],
            }
            for label in SENTIMENT_ORDER
        },
        "emotion_sentiment_matrix": {
            emo: {sent: cross_tab[emo].get(sent, 0) for sent in SENTIMENT_ORDER}
            for emo in EMOTION_ORDER
        },
        "utterance_word_length_stats": summarize_numeric_series(word_lengths),
        "clip_integrity": {
            "checked": clips_root is not None,
            "missing_clip_count": len(missing_clips),
            "corrupted_clip_count": len(corrupted_clips),
            "missing_clips_sample": missing_clips[:20],
            "corrupted_clips_sample": corrupted_clips[:20],
            "duration_seconds_stats": summarize_numeric_series(clip_durations),
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile MELD split distributions, utterance lengths, and clip integrity."
    )
    parser.add_argument("--csv", required=True, help="Path to MELD split CSV file")
    parser.add_argument("--video-dir", default=None, help="Optional path to MELD mp4 directory")
    parser.add_argument(
        "--probe-videos",
        action="store_true",
        help="Run ffprobe on each clip to validate readability and duration stats",
    )
    parser.add_argument("--output", default="dataset_profile.json", help="Output JSON report path")
    args = parser.parse_args()

    report = profile_meld_split(
        csv_path=args.csv,
        video_dir=args.video_dir,
        probe_videos=args.probe_videos,
        output_path=args.output,
    )
    print(f"Profiled {report['total_utterances']} utterances -> {args.output}")
    print(f"Utterance word stats: {report['utterance_word_length_stats']}")


if __name__ == "__main__":
    main()
