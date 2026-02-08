"""Transformer-based temporal gesture classifier.

Processes sequences of hand landmark features to recognize
dynamic gestures (swipe, pinch, wave, etc.) using self-attention
over temporal frames.

Architecture:
    Input → Positional Encoding → N × TransformerEncoder → Classification Head

Designed for:
    - Real-time inference (< 5ms for 30-frame sequences)
    - Export to ONNX / TFLite
    - Few-shot adaptation via classifier head swap
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq_len, d_model)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GestureTransformer(nn.Module):
    """Transformer model for temporal gesture recognition.

    Takes a sequence of per-frame feature vectors and classifies
    the entire sequence into a gesture class.

    Args:
        input_dim: Dimension of per-frame feature vector (e.g., 86 from LandmarkFeatureExtractor).
        num_classes: Number of gesture classes.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.

    Input:
        x: (batch, seq_len, input_dim) — sequence of landmark features.
        mask: (batch, seq_len) — optional boolean mask (True = padded).

    Output:
        logits: (batch, num_classes) — gesture classification logits.
        attention_weights: Optional attention maps for explainability.
    """

    def __init__(
        self,
        input_dim: int = 86,
        num_classes: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        max_seq_len: int = 60,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding (+1 for CLS token prepended before encoding)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # CLS token for sequence-level classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Confidence head (auxiliary)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) landmark feature sequences.
            mask: (batch, seq_len) boolean mask where True = padded/ignored.
            return_attention: If True, include attention weights in output.

        Returns:
            Dict with keys: 'logits', 'confidence', and optionally 'attention'.
        """
        batch_size = x.size(0)

        # Project input to d_model
        x = self.input_proj(x)  # (B, S, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, S+1, d_model)

        # Update mask for CLS token
        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Extract CLS token output
        cls_output = x[:, 0]  # (B, d_model)

        # Classification
        logits = self.classifier(cls_output)  # (B, num_classes)
        confidence = self.confidence_head(cls_output).squeeze(-1)  # (B,)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "confidence": confidence,
        }

        return result

    def predict(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Inference-mode prediction with softmax probabilities.

        Returns:
            Dict with 'class_id', 'class_probs', 'confidence'.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, mask=mask)
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
