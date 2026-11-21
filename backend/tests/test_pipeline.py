from pathlib import Path
import sys
import unittest

import torch

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from deployment.video_chunker import merge_chunk_predictions
from training.models import AudioEncoder, MultimodalSentimentModel


class TestMoodLensPipeline(unittest.TestCase):
    def test_audio_encoder_output_shape(self) -> None:
        encoder = AudioEncoder()
        encoder.eval()

        dummy_audio = torch.randn(2, 1, 64, 300)
        with torch.inference_mode():
            features = encoder(dummy_audio)

        self.assertEqual(tuple(features.shape), (2, 128))
        self.assertFalse(torch.isnan(features).any())

    def test_multimodal_forward_pass_shapes(self) -> None:
        model = MultimodalSentimentModel()
        model.eval()

        batch_size = 2
        text_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 128)),
            "attention_mask": torch.ones((batch_size, 128), dtype=torch.long),
        }
        video_frames = torch.randn(batch_size, 30, 3, 224, 224)
        audio_features = torch.randn(batch_size, 1, 64, 300)

        with torch.inference_mode():
            outputs = model(text_inputs, video_frames, audio_features)

        self.assertIn("emotions", outputs)
        self.assertIn("sentiments", outputs)
        self.assertEqual(tuple(outputs["emotions"].shape), (batch_size, 7))
        self.assertEqual(tuple(outputs["sentiments"].shape), (batch_size, 3))
        self.assertFalse(torch.isnan(outputs["emotions"]).any())
        self.assertFalse(torch.isnan(outputs["sentiments"]).any())

    def test_merge_chunk_predictions_offsets(self) -> None:
        chunk_outputs = [
            {
                "start_offset": 0.0,
                "utterances": [
                    {"start": 1.2, "end": 3.5, "emotion": "joy", "sentiment": "positive"}
                ],
            },
            {
                "start_offset": 60.0,
                "utterances": [
                    {"start": 2.0, "end": 5.4, "emotion": "joy", "sentiment": "positive"},
                    {"start": 6.0, "end": 8.1, "emotion": "neutral", "sentiment": "neutral"},
                ],
            },
        ]

        merged = merge_chunk_predictions(chunk_outputs)

        self.assertEqual(len(merged["utterances"]), 3)
        self.assertEqual(merged["utterances"][1]["start"], 62.0)
        self.assertEqual(merged["utterances"][1]["end"], 65.4)
        self.assertEqual(merged["overall_emotion"], "joy")
        self.assertEqual(merged["overall_sentiment"], "positive")


if __name__ == "__main__":
    unittest.main()