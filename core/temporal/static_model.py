"""Lightweight MLP for single-frame (static) gesture classification.

Used when temporal data is not available — e.g., single image input,
or as a fast baseline before the full Transformer pipeline.

Architecture:
    Input → [Linear → GELU → BatchNorm → Dropout] × N → Linear → logits

Designed for:
    - Single-frame gesture detection (no sequence needed)
    - Baseline comparison against Transformer model
    - Few-shot adaptation (freeze backbone, retrain head)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticGestureClassifier(nn.Module):
    """MLP classifier for single-frame gesture recognition.

    Takes a single feature vector (e.g., 86-dim from LandmarkFeatureExtractor)
    and classifies it into a gesture class. No temporal modeling.

    Args:
        input_dim: Dimension of input feature vector.
        num_classes: Number of gesture classes.
        hidden_dims: Hidden layer dimensions.
        dropout: Dropout rate.

    Input:
        x: (batch, input_dim) — single-frame feature vectors.

    Output:
        Dict with 'logits' and 'confidence'.
    """

    def __init__(
        self,
        input_dim: int = 86,
        num_classes: int = 10,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.GELU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid(),
        )

        # Build a sub-network that outputs the penultimate features
        # for the confidence head
        self._feature_layers = nn.Sequential(*layers[:-1])
        self._classifier_head = layers[-1]

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for ReLU-family activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, input_dim) single-frame features.

        Returns:
            Dict with 'logits' (batch, num_classes) and 'confidence' (batch,).
        """
        features = self._feature_layers(x)       # (B, last_hidden)
        logits = self._classifier_head(features)  # (B, num_classes)
        confidence = self.confidence_head(features).squeeze(-1)  # (B,)
        return {"logits": logits, "confidence": confidence}

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference-mode prediction with softmax probabilities."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output["logits"], dim=-1)
            class_ids = torch.argmax(probs, dim=-1)
            return {
                "class_id": class_ids,
                "class_probs": probs,
                "confidence": output["confidence"],
            }

    def get_model_size_mb(self) -> float:
        """Return model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_size / (1024 * 1024)
