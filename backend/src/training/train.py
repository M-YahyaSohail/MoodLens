import argparse
import os
import sys

import torch
import torchaudio
from src.training.install_ffmpeg import ensure_ffmpeg
from src.training.meld_dataset import prepare_dataloaders
from src.training.models import MultimodalSentimentModel, MultimodalTrainer
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # Data directories (relative to the backend project root)
    parser.add_argument("--train-dir", type=str, default="dataset/train")
    parser.add_argument("--val-dir", type=str, default="dataset/dev")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--model-dir", type=str, default=".")

    return parser.parse_args()


def main():
    if not ensure_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, "train_sent_emo.csv"),
        train_video_dir=os.path.join(args.train_dir, "train_splits"),
        dev_csv=os.path.join(args.val_dir, "dev_sent_emo.csv"),
        dev_video_dir=os.path.join(args.val_dir, "dev_splits_complete"),
        test_csv=os.path.join(args.test_dir, "test_sent_emo.csv"),
        test_video_dir=os.path.join(args.test_dir, "output_repeated_splits_test"),
        batch_size=args.batch_size,
    )

    print(
        f"""Training DSV path: {os.path.join(args.train_dir, "train_sent_emo.csv")}"""
    )
    print(
        f"""Training video directory: {os.path.join(args.train_dir, "train_splits")}"""
    )

    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float("inf")

    metrics_data = {"train_losses": [], "val_losses": [], "epochs": []}

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # Print metrics for monitoring
        print(
            f"""Epoch {epoch + 1}/{args.epochs}: train_loss={
                train_loss["total"]:.4f
            }, val_loss={val_loss["total"]:.4f}, val_emotion_acc={
                val_metrics["emotion_accuracy"]:.4f
            }, val_sentiment_acc={val_metrics["sentiment_accuracy"]:.4f}"""
        )

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

    # After training is complete, evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_loss["total"]

    print(
        f"""Test: loss={test_loss["total"]:.4f}, emotion_acc={
            test_metrics["emotion_accuracy"]:.4f
        }, sentiment_acc={test_metrics["sentiment_accuracy"]:.4f}"""
    )


if __name__ == "__main__":
    main()
