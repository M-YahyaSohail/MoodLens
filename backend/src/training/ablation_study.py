import argparse
import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, default_collate

try:
    from src.training.meld_dataset import MELDDataset
    from src.training.models import MultimodalSentimentModel
except ImportError:
    from meld_dataset import MELDDataset
    from models import MultimodalSentimentModel


MODALITY_CONFIGS: dict[str, tuple[str, ...]] = {
    "full_multimodal": ("text", "video", "audio"),
    "text_only": ("text",),
    "video_only": ("video",),
    "audio_only": ("audio",),
    "text_video": ("text", "video"),
    "text_audio": ("text", "audio"),
    "video_audio": ("video", "audio"),
}


def _safe_collate(batch: list[Any]) -> dict[str, Any] | None:
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    return default_collate(valid)


@torch.inference_mode()
def run_ablation_study(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    video_dir: str | Path,
    batch_size: int = 8,
    output_path: str | Path = "ablation_report.json",
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

    predictions: dict[str, dict[str, list[int]]] = {
        name: {"emotions": [], "sentiments": []} for name in MODALITY_CONFIGS
    }
    emo_targets: list[int] = []
    sent_targets: list[int] = []

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

        zero_text = torch.zeros_like(text_feat)
        zero_video = torch.zeros_like(video_feat)
        zero_audio = torch.zeros_like(audio_feat)

        emo_targets.extend(batch["emotion_label"].cpu().tolist())
        sent_targets.extend(batch["sentiment_label"].cpu().tolist())

        for config_name, active in MODALITY_CONFIGS.items():
            t = text_feat if "text" in active else zero_text
            v = video_feat if "video" in active else zero_video
            a = audio_feat if "audio" in active else zero_audio

            fused = model.fusion_layer(torch.cat([t, v, a], dim=1))
            emo_logits = model.emotion_classifier(fused)
            sent_logits = model.sentiment_classifier(fused)

            predictions[config_name]["emotions"].extend(emo_logits.argmax(dim=1).cpu().tolist())
            predictions[config_name]["sentiments"].extend(sent_logits.argmax(dim=1).cpu().tolist())

    results: dict[str, Any] = {}
    for config_name, preds in predictions.items():
        results[config_name] = {
            "active_modalities": list(MODALITY_CONFIGS[config_name]),
            "emotion_accuracy": round(float(accuracy_score(emo_targets, preds["emotions"])), 4),
            "emotion_macro_f1": round(
                float(f1_score(emo_targets, preds["emotions"], average="macro", zero_division=0)),
                4,
            ),
            "sentiment_accuracy": round(
                float(accuracy_score(sent_targets, preds["sentiments"])), 4
            ),
            "sentiment_macro_f1": round(
                float(
                    f1_score(sent_targets, preds["sentiments"], average="macro", zero_division=0)
                ),
                4,
            ),
        }

    baseline_emo_f1 = results["full_multimodal"]["emotion_macro_f1"]
    baseline_sent_f1 = results["full_multimodal"]["sentiment_macro_f1"]
    for _, metrics in results.items():
        metrics["emotion_f1_delta_vs_full"] = round(
            metrics["emotion_macro_f1"] - baseline_emo_f1, 4
        )
        metrics["sentiment_f1_delta_vs_full"] = round(
            metrics["sentiment_macro_f1"] - baseline_sent_f1, 4
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "num_samples": len(emo_targets),
        "ablation_results": results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run modality ablation benchmark on MultimodalSentimentModel."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pth file")
    parser.add_argument("--csv", required=True, help="Path to MELD split CSV file")
    parser.add_argument("--video-dir", required=True, help="Path to MELD video clips directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--output", default="ablation_report.json", help="Output JSON report path")
    args = parser.parse_args()

    report = run_ablation_study(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        video_dir=args.video_dir,
        batch_size=args.batch_size,
        output_path=args.output,
    )

    print(f"Ablation benchmark complete ({report['num_samples']} samples) -> {args.output}")
    for name, metrics in report["ablation_results"].items():
        print(
            f"{name:16s} | Emo F1: {metrics['emotion_macro_f1']:.4f} "
            f"({metrics['emotion_f1_delta_vs_full']:+.4f}) | "
            f"Sent F1: {metrics['sentiment_macro_f1']:.4f} "
            f"({metrics['sentiment_f1_delta_vs_full']:+.4f})"
        )


if __name__ == "__main__":
    main()
