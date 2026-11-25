from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


EMOTION_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
SENTIMENT_LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class UtteranceMetrics:
    index: int
    start_time: float
    end_time: float
    duration: float
    text: str
    top_emotion: str
    emotion_confidence: float
    top_sentiment: str
    sentiment_confidence: float
    valence_score: float


def _extract_top_prediction(
    predictions: Any,
    fallback_label: str = "neutral",
) -> tuple[str, float, dict[str, float]]:
    if isinstance(predictions, str) and predictions:
        return predictions.lower(), 1.0, {predictions.lower(): 1.0}

    if isinstance(predictions, list) and predictions:
        score_map: dict[str, float] = {}
        for entry in predictions:
            if isinstance(entry, dict) and "label" in entry:
                label = str(entry["label"]).lower()
                conf = float(entry.get("confidence", 0.0))
                score_map[label] = conf

        if score_map:
            best_label = max(score_map, key=lambda key: score_map[key])
            return best_label, round(score_map[best_label], 4), score_map

    return fallback_label, 0.0, {}


def _compute_valence(sentiment_scores: dict[str, float], top_sentiment: str) -> float:
    if "positive" in sentiment_scores or "negative" in sentiment_scores:
        pos = sentiment_scores.get("positive", 0.0)
        neg = sentiment_scores.get("negative", 0.0)
        return round(pos - neg, 4)

    polarity_defaults = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    return polarity_defaults.get(top_sentiment, 0.0)


def parse_utterance_metrics(utterances: list[dict[str, Any]]) -> list[UtteranceMetrics]:
    parsed: list[UtteranceMetrics] = []

    for idx, item in enumerate(utterances):
        start = float(item.get("start_time", item.get("start", 0.0)))
        end = float(item.get("end_time", item.get("end", start)))
        duration = max(0.0, round(end - start, 3))
        text = str(item.get("text", "")).strip()

        raw_emotions = item.get("emotions", item.get("emotion"))
        raw_sentiments = item.get("sentiments", item.get("sentiment"))

        top_emo, emo_conf, _ = _extract_top_prediction(raw_emotions)
        top_sent, sent_conf, sent_map = _extract_top_prediction(raw_sentiments)
        valence = _compute_valence(sent_map, top_sent)

        parsed.append(
            UtteranceMetrics(
                index=idx,
                start_time=round(start, 3),
                end_time=round(end, 3),
                duration=duration,
                text=text,
                top_emotion=top_emo,
                emotion_confidence=emo_conf,
                top_sentiment=top_sent,
                sentiment_confidence=sent_conf,
                valence_score=valence,
            )
        )

    return parsed


def detect_mood_shifts(
    metrics: list[UtteranceMetrics],
    min_confidence: float = 0.25,
) -> list[dict[str, Any]]:
    shifts: list[dict[str, Any]] = []

    for prev, curr in zip(metrics, metrics[1:]):
        if prev.emotion_confidence < min_confidence or curr.emotion_confidence < min_confidence:
            continue

        emotion_changed = prev.top_emotion != curr.top_emotion
        sentiment_changed = prev.top_sentiment != curr.top_sentiment

        if emotion_changed or sentiment_changed:
            shifts.append(
                {
                    "timestamp": curr.start_time,
                    "from_utterance_index": prev.index,
                    "to_utterance_index": curr.index,
                    "from_emotion": prev.top_emotion,
                    "to_emotion": curr.top_emotion,
                    "from_sentiment": prev.top_sentiment,
                    "to_sentiment": curr.top_sentiment,
                    "valence_delta": round(curr.valence_score - prev.valence_score, 4),
                    "trigger_text": curr.text,
                }
            )

    return shifts


def summarize_video_analytics(
    utterances: list[dict[str, Any]],
    min_shift_confidence: float = 0.25,
) -> dict[str, Any]:
    metrics = parse_utterance_metrics(utterances)
    if not metrics:
        return {
            "utterance_count": 0,
            "total_speech_duration": 0.0,
            "dominant_emotion": "neutral",
            "dominant_sentiment": "neutral",
            "mean_valence": 0.0,
            "emotion_share_by_count": {},
            "emotion_share_by_duration": {},
            "sentiment_share_by_count": {},
            "peak_emotional_utterance": None,
            "mood_shifts": [],
            "valence_timeline": [],
        }

    total_count = len(metrics)
    total_duration = round(sum(m.duration for m in metrics), 3)

    emo_counts: Counter[str] = Counter(m.top_emotion for m in metrics)
    sent_counts: Counter[str] = Counter(m.top_sentiment for m in metrics)

    emo_durations: dict[str, float] = {label: 0.0 for label in EMOTION_LABELS}
    for m in metrics:
        emo_durations[m.top_emotion] = round(
            emo_durations.get(m.top_emotion, 0.0) + m.duration, 3
        )

    dominant_emotion = emo_counts.most_common(1)[0][0]
    dominant_sentiment = sent_counts.most_common(1)[0][0]
    mean_valence = round(sum(m.valence_score for m in metrics) / total_count, 4)

    non_neutral = [m for m in metrics if m.top_emotion != "neutral"]
    peak_candidates = non_neutral if non_neutral else metrics
    peak = max(peak_candidates, key=lambda m: m.emotion_confidence)

    emotion_share_duration = (
        {
            label: round(dur / total_duration, 4)
            for label, dur in emo_durations.items()
            if dur > 0
        }
        if total_duration > 0
        else {}
    )

    return {
        "utterance_count": total_count,
        "total_speech_duration": total_duration,
        "dominant_emotion": dominant_emotion,
        "dominant_sentiment": dominant_sentiment,
        "mean_valence": mean_valence,
        "emotion_share_by_count": {
            label: round(count / total_count, 4) for label, count in emo_counts.items()
        },
        "emotion_share_by_duration": emotion_share_duration,
        "sentiment_share_by_count": {
            label: round(count / total_count, 4) for label, count in sent_counts.items()
        },
        "peak_emotional_utterance": {
            "index": peak.index,
            "start_time": peak.start_time,
            "end_time": peak.end_time,
            "emotion": peak.top_emotion,
            "confidence": peak.emotion_confidence,
            "text": peak.text,
        },
        "mood_shifts": detect_mood_shifts(metrics, min_confidence=min_shift_confidence),
        "valence_timeline": [
            {
                "index": m.index,
                "start_time": m.start_time,
                "end_time": m.end_time,
                "emotion": m.top_emotion,
                "sentiment": m.top_sentiment,
                "valence": m.valence_score,
            }
            for m in metrics
        ],
    }


def export_utterances_csv(
    utterances: list[dict[str, Any]],
    output_csv_path: str | Path,
) -> Path:
    metrics = parse_utterance_metrics(utterances)
    destination = Path(output_csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "start_time",
        "end_time",
        "duration",
        "top_emotion",
        "emotion_confidence",
        "top_sentiment",
        "sentiment_confidence",
        "valence_score",
        "text",
    ]

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "index": item.index,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "duration": item.duration,
                    "top_emotion": item.top_emotion,
                    "emotion_confidence": item.emotion_confidence,
                    "top_sentiment": item.top_sentiment,
                    "sentiment_confidence": item.sentiment_confidence,
                    "valence_score": item.valence_score,
                    "text": item.text,
                }
            )

    return destination


def export_analytics_json(
    utterances: list[dict[str, Any]],
    output_json_path: str | Path,
) -> dict[str, Any]:
    summary = summarize_video_analytics(utterances)
    destination = Path(output_json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary