from typing import Any

import torch
import torch.nn as nn
from torchvision import models as vision_models
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert: BertModel = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection: nn.Linear = nn.Linear(768, 128)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Extract BERT embeddings
        outputs: Any = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output: torch.Tensor = outputs.pooler_output

        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts: int = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, frames, channels, height, width]
        # -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_layers: nn.Sequential = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)

        features: torch.Tensor = self.conv_layers(x)
        # Features output: [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Encoders
        self.text_encoder: TextEncoder = TextEncoder()
        self.video_encoder: VideoEncoder = VideoEncoder()
        self.audio_encoder: AudioEncoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer: nn.Sequential = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification heads
        self.emotion_classifier: nn.Sequential = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),  # Sadness, anger
        )

        self.sentiment_classifier: nn.Sequential = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # Negative, positive, neutral
        )

    def forward(
        self,
        text_inputs: dict[str, torch.Tensor],
        video_frames: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_features: torch.Tensor = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
        )
        video_features: torch.Tensor = self.video_encoder(video_frames)
        audio_encoded: torch.Tensor = self.audio_encoder(audio_features)

        # Concatenate multimodal features
        combined_features: torch.Tensor = torch.cat(
            [text_features, video_features, audio_encoded], dim=1
        )  # [batch_size, 128 * 3]

        fused_features: torch.Tensor = self.fusion_layer(combined_features)

        emotion_output: torch.Tensor = self.emotion_classifier(fused_features)
        sentiment_output: torch.Tensor = self.sentiment_classifier(fused_features)

        return {"emotions": emotion_output, "sentiments": sentiment_output}
