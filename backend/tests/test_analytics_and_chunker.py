import csv
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"
for path in (BACKEND_ROOT, BACKEND_SRC, BACKEND_SRC / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from deployment.analytics_aggregator import (
    export_analytics_json,
    export_utterances_csv,
    summarize_video_analytics,
)
from deployment.video_chunker import (
    get_video_duration,
    merge_chunk_predictions,
    split_video_into_chunks,
)
from training.dataset_profiler import (
    compute_inverse_weights,
    compute_percentile,
    profile_meld_split,
)
from training.error_analysis import check_head_consistency


class TestAnalyticsAndChunker(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_utterances = [
            {
                "start_time": 0.0,
                "end_time": 2.5,
                "text": "I cannot believe we finally got the grant!",
                "emotions": [
                    {"label": "joy", "confidence": 0.78},
                    {"label": "surprise", "confidence": 0.15},
                ],
                "sentiments": [
                    {"label": "positive", "confidence": 0.85},
                    {"label": "neutral", "confidence": 0.10},
                    {"label": "negative", "confidence": 0.05},
                ],
            },
            {
                "start_time": 2.5,
                "end_time": 6.5,
                "text": "Wait, why did the server just crash?",
                "emotions": [
                    {"label": "fear", "confidence": 0.64},
                    {"label": "anger", "confidence": 0.22},
                ],
                "sentiments": [
                    {"label": "negative", "confidence": 0.74},
                    {"label": "neutral", "confidence": 0.20},
                    {"label": "positive", "confidence": 0.06},
                ],
            },
            {
                "start_time": 6.5,
                "end_time": 8.5,
                "text": "Okay, the backup instance is online.",
                "emotions": [
                    {"label": "neutral", "confidence": 0.71},
                    {"label": "joy", "confidence": 0.19},
                ],
                "sentiments": [
                    {"label": "neutral", "confidence": 0.68},
                    {"label": "positive", "confidence": 0.22},
                    {"label": "negative", "confidence": 0.10},
                ],
            },
        ]

    def test_summarize_video_analytics_metrics(self) -> None:
        summary = summarize_video_analytics(self.sample_utterances)

        self.assertEqual(summary["utterance_count"], 3)
        self.assertEqual(summary["total_speech_duration"], 8.5)
        self.assertEqual(summary["peak_emotional_utterance"]["emotion"], "joy")
        self.assertEqual(summary["peak_emotional_utterance"]["index"], 0)
        self.assertEqual(len(summary["mood_shifts"]), 2)
        self.assertEqual(summary["mood_shifts"][0]["from_emotion"], "joy")
        self.assertEqual(summary["mood_shifts"][0]["to_emotion"], "fear")
        self.assertAlmostEqual(summary["mean_valence"], 0.08, places=4)

    def test_empty_utterances_summary(self) -> None:
        summary = summarize_video_analytics([])

        self.assertEqual(summary["utterance_count"], 0)
        self.assertEqual(summary["dominant_emotion"], "neutral")
        self.assertEqual(summary["dominant_sentiment"], "neutral")
        self.assertIsNone(summary["peak_emotional_utterance"])

    def test_export_csv_and_json_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_out = export_utterances_csv(self.sample_utterances, tmp_path / "utterances.csv")
            json_out = tmp_path / "analytics.json"
            summary = export_analytics_json(self.sample_utterances, json_out)

            self.assertTrue(csv_out.is_file())
            self.assertTrue(json_out.is_file())

            with csv_out.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["top_emotion"], "fear")

            loaded_json = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(loaded_json["utterance_count"], summary["utterance_count"])

    def test_chunker_validation_and_empty_merge(self) -> None:
        with self.assertRaises(FileNotFoundError):
            get_video_duration("non_existent_clip_12345.mp4")

        with self.assertRaises(ValueError):
            split_video_into_chunks("dummy.mp4", "chunks", max_chunk_duration=0.0)

        merged = merge_chunk_predictions([])
        self.assertEqual(merged["utterances"], [])
        self.assertEqual(merged["overall_emotion"], "neutral")

    def test_dataset_profiler_and_head_consistency(self) -> None:
        self.assertEqual(compute_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0), 3.0)

        weights = compute_inverse_weights(Counter({"joy": 2, "neutral": 8}), ["joy", "neutral"])
        self.assertAlmostEqual(weights["joy"], 0.8, places=4)
        self.assertAlmostEqual(weights["neutral"], 0.2, places=4)

        self.assertTrue(check_head_consistency("joy", "positive"))
        self.assertTrue(check_head_consistency("anger", "negative"))
        self.assertFalse(check_head_consistency("joy", "negative"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sample_csv = tmp_path / "sample_meld.csv"
            with sample_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Utterance",
                        "Emotion",
                        "Sentiment",
                        "Speaker",
                        "Dialogue_ID",
                        "Utterance_ID",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Utterance": "That is awesome news!",
                        "Emotion": "joy",
                        "Sentiment": "positive",
                        "Speaker": "Monica",
                        "Dialogue_ID": "0",
                        "Utterance_ID": "0",
                    }
                )
                writer.writerow(
                    {
                        "Utterance": "I do not know what to say.",
                        "Emotion": "sadness",
                        "Sentiment": "negative",
                        "Speaker": "Ross",
                        "Dialogue_ID": "0",
                        "Utterance_ID": "1",
                    }
                )

            profile = profile_meld_split(
                csv_path=sample_csv,
                output_path=tmp_path / "profile.json",
            )
            self.assertEqual(profile["total_utterances"], 2)
            self.assertEqual(profile["unique_speakers"], 2)
            self.assertEqual(profile["emotion_distribution"]["joy"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
