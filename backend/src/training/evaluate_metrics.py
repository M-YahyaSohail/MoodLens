import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader

try:
    from src.training.meld_dataset import MELDDataset
    from src.training.models import MultimodalSentimentModel
except ImportError:
    from meld_dataset import MELDDataset
    from models import MultimodalSentimentModel


EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def build_task_report(
    y_true: list[int],
    y_pred: list[int],
    target_names: list[str],
) -> dict[str, Any]:
    labels = list(range(len(target_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class": {
            name: {
                "precision": round(float(report[name]["precision"]), 4),
                "recall": round(float(report[name]["recall"]), 4),
                "f1": round(float(report[name]["f1-score"]), 4),
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
        "confusion_matrix": matrix,
    }


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    video_dir: str | Path,
    batch_size: int = 8,
    output_path: str | Path = "eval_report.json",
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MELDDataset(str(csv_path), str(video_dir))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MultimodalSentimentModel().to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    emo_preds: list[int] = []
    emo_targets: list[int] = []
    sent_preds: list[int] = []
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

        outputs = model(text_inputs, video_frames, audio_features)

        emo_preds.extend(outputs["emotions"].argmax(dim=1).cpu().tolist())
        emo_targets.extend(batch["emotion_label"].cpu().tolist())

        sent_preds.extend(outputs["sentiments"].argmax(dim=1).cpu().tolist())
        sent_targets.extend(batch["sentiment_label"].cpu().tolist())

    report = {
        "checkpoint": str(checkpoint_path),
        "num_samples": len(emo_targets),
        "emotion": build_task_report(emo_targets, emo_preds, EMOTION_LABELS),
        "sentiment": build_task_report(sent_targets, sent_preds, SENTIMENT_LABELS),
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MultimodalSentimentModel per-class F1 and confusion matrices."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pth file")
    parser.add_argument("--csv", required=True, help="Path to MELD split CSV file")
    parser.add_argument("--video-dir", required=True, help="Path to MELD video clips directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--output", default="eval_report.json", help="Output JSON report path")
    args = parser.parse_args()

    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        video_dir=args.video_dir,
        batch_size=args.batch_size,
        output_path=args.output,
    )

    print(f"Evaluated {report['num_samples']} samples -> {args.output}")
    print(
        f"Emotion   | Accuracy: {report['emotion']['accuracy']:.4f} | "
        f"Macro-F1: {report['emotion']['macro_f1']:.4f}"
    )
    print(
        f"Sentiment | Accuracy: {report['sentiment']['accuracy']:.4f} | "
        f"Macro-F1: {report['sentiment']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()