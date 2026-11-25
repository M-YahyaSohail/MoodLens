import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, default_collate

try:
    from src.training.meld_dataset import MELDDataset
    from src.training.models import MultimodalSentimentModel
except ImportError:
    from meld_dataset import MELDDataset
    from models import MultimodalSentimentModel


EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

EXPECTED_EMOTION_POLARITY: dict[str, set[str]] = {
    "anger": {"negative"},
    "disgust": {"negative"},
    "fear": {"negative"},
    "sadness": {"negative"},
    "joy": {"positive"},
    "neutral": {"neutral"},
    "surprise": {"negative", "neutral", "positive"},
}


def _safe_collate(batch: list[Any]) -> dict[str, Any] | None:
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    return default_collate(valid)


def check_head_consistency(emotion_label: str, sentiment_label: str) -> bool:
    allowed = EXPECTED_EMOTION_POLARITY.get(emotion_label, set())
    return sentiment_label in allowed


@torch.inference_mode()
def analyze_checkpoint_errors(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    video_dir: str | Path,
    batch_size: int = 8,
    top_k: int = 25,
    output_path: str | Path = "error_analysis.json",
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MELDDataset(str(csv_path), str(video_dir))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_safe_collate,
    )

    model = MultimodalSentimentModel().to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_samples = 0
    consistent_heads = 0
    modality_rescues: list[dict[str, Any]] = []
    high_conf_emotion_errors: list[dict[str, Any]] = []
    high_conf_sentiment_errors: list[dict[str, Any]] = []

    for batch in loader:
        if batch is None:
            continue

        text_inputs = {
            "input_ids": batch["text_inputs"]["input_ids"].to(device),
            "attention_mask": batch["text_inputs"]["attention_mask"].to(device),
        }
        video_frames = batch["video_frames"].to(device)
        audio_features = batch["audio_features"].to(device)

        text_feat = model.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
        )
        video_feat = model.video_encoder(video_frames)
        audio_feat = model.audio_encoder(audio_features)

        full_fused = model.fusion_layer(
            torch.cat([text_feat, video_feat, audio_feat], dim=1)
        )
        full_emo_probs = torch.softmax(model.emotion_classifier(full_fused), dim=1)
        full_sent_probs = torch.softmax(model.sentiment_classifier(full_fused), dim=1)

        text_only_fused = model.fusion_layer(
            torch.cat(
                [text_feat, torch.zeros_like(video_feat), torch.zeros_like(audio_feat)],
                dim=1,
            )
        )
        text_emo_preds = model.emotion_classifier(text_only_fused).argmax(dim=1)
        text_sent_preds = model.sentiment_classifier(text_only_fused).argmax(dim=1)

        emo_targets = batch["emotion_label"].to(device)
        sent_targets = batch["sentiment_label"].to(device)

        for idx in range(emo_targets.size(0)):
            sample_id = total_samples
            total_samples += 1

            true_emo = int(emo_targets[idx].item())
            true_sent = int(sent_targets[idx].item())

            pred_emo = int(full_emo_probs[idx].argmax().item())
            pred_emo_conf = float(full_emo_probs[idx, pred_emo].item())

            pred_sent = int(full_sent_probs[idx].argmax().item())
            pred_sent_conf = float(full_sent_probs[idx, pred_sent].item())

            txt_emo = int(text_emo_preds[idx].item())
            txt_sent = int(text_sent_preds[idx].item())

            if check_head_consistency(EMOTION_LABELS[pred_emo], SENTIMENT_LABELS[pred_sent]):
                consistent_heads += 1

            if (txt_emo != true_emo and pred_emo == true_emo) or (
                txt_sent != true_sent and pred_sent == true_sent
            ):
                modality_rescues.append(
                    {
                        "sample_index": sample_id,
                        "true_emotion": EMOTION_LABELS[true_emo],
                        "text_only_emotion": EMOTION_LABELS[txt_emo],
                        "multimodal_emotion": EMOTION_LABELS[pred_emo],
                        "true_sentiment": SENTIMENT_LABELS[true_sent],
                        "text_only_sentiment": SENTIMENT_LABELS[txt_sent],
                        "multimodal_sentiment": SENTIMENT_LABELS[pred_sent],
                    }
                )

            if pred_emo != true_emo:
                high_conf_emotion_errors.append(
                    {
                        "sample_index": sample_id,
                        "true_emotion": EMOTION_LABELS[true_emo],
                        "predicted_emotion": EMOTION_LABELS[pred_emo],
                        "confidence": round(pred_emo_conf, 4),
                    }
                )

            if pred_sent != true_sent:
                high_conf_sentiment_errors.append(
                    {
                        "sample_index": sample_id,
                        "true_sentiment": SENTIMENT_LABELS[true_sent],
                        "predicted_sentiment": SENTIMENT_LABELS[pred_sent],
                        "confidence": round(pred_sent_conf, 4),
                    }
                )

    high_conf_emotion_errors.sort(key=lambda item: item["confidence"], reverse=True)
    high_conf_sentiment_errors.sort(key=lambda item: item["confidence"], reverse=True)

    consistency_rate = (
        round(consistent_heads / total_samples, 4) if total_samples > 0 else 0.0
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "total_samples": total_samples,
        "head_consistency_rate": consistency_rate,
        "modality_rescue_count": len(modality_rescues),
        "modality_rescues_sample": modality_rescues[:top_k],
        "top_emotion_errors": high_conf_emotion_errors[:top_k],
        "top_sentiment_errors": high_conf_sentiment_errors[:top_k],
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run modality-conflict and high-confidence error analysis."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--csv", required=True, help="Path to MELD split CSV file")
    parser.add_argument("--video-dir", required=True, help="Path to MELD video clips directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--top-k", type=int, default=25, help="Max records per category")
    parser.add_argument("--output", default="error_analysis.json", help="Output JSON path")
    args = parser.parse_args()

    report = analyze_checkpoint_errors(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        video_dir=args.video_dir,
        batch_size=args.batch_size,
        top_k=args.top_k,
        output_path=args.output,
    )
    print(f"Analyzed {report['total_samples']} samples -> {args.output}")
    print(f"Cross-head consistency rate: {report['head_consistency_rate']:.4f}")
    print(f"Multimodal rescues over text-only: {report['modality_rescue_count']}")


if __name__ == "__main__":
    main()