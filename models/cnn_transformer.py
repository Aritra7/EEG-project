"""
CNN + Transformer + Multi-Head Architecture (Task 1 & 2)
=========================================================
Three-stage architecture from Section 3 of the project spec:
  Stage 1: 1D CNN feature extractor (per-channel temporal convolutions)
  Stage 2: Shared Transformer backbone (cross-electrode attention)
  Stage 3: Subject-specific prediction heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNNBlock(nn.Module):
    """Single 1D conv block: Conv1d → BN → GELU → Dropout"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))


class CNNFeatureExtractor(nn.Module):
    """
    Stage 1: Per-channel temporal CNN.

    Input:  (B, 122, 500) — raw EEG
    Output: (B, 122, embed_dim) — per-electrode temporal embeddings
    """

    def __init__(self, embed_dim=64, num_layers=2, kernel_size=25, dropout=0.1):
        super().__init__()

        # Build conv layers: 1 → mid → ... → embed_dim
        channels = [1] + [embed_dim // (2 ** max(0, num_layers - 1 - i))
                         for i in range(num_layers)]
        # Ensure last channel is embed_dim
        channels[-1] = embed_dim

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                TemporalCNNBlock(channels[i], channels[i + 1],
                                 kernel_size=kernel_size, dropout=dropout)
            )

        # Temporal pooling → collapse time dim via mean pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:  x: (B, 122, 500)
        Returns:   (B, 122, embed_dim)
        """
        B, C, T = x.shape  # B, 122, 500

        # Treat each electrode independently: reshape to (B*122, 1, 500)
        x = x.view(B * C, 1, T)

        # Apply conv layers
        for conv in self.conv_layers:
            x = conv(x)  # (B*122, embed_dim, T')

        # Mean pool over time → (B*122, embed_dim, 1) → (B*122, embed_dim)
        x = self.pool(x).squeeze(-1)

        # Reshape back: (B, 122, embed_dim)
        x = x.view(B, C, -1)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for electrode positions."""

    def __init__(self, num_positions, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_positions, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embed


class TransformerBackbone(nn.Module):
    """
    Stage 2: Shared Transformer encoder.

    Input:  (B, 122, embed_dim) — per-electrode features
    Output: (B, embed_dim) — global EEG representation (CLS token or mean pool)
    """

    def __init__(self, embed_dim=64, num_heads=8, num_layers=4,
                 feedforward_dim=256, dropout=0.1, num_electrodes=122):
        super().__init__()

        self.pos_encoding = PositionalEncoding(num_electrodes, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:  x: (B, 122, embed_dim)
        Returns:   (B, embed_dim) — global representation
        """
        B = x.size(0)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 123, embed_dim)

        # Transformer encoding
        x = self.encoder(x)  # (B, 123, embed_dim)
        x = self.norm(x)

        # Use CLS token as global representation
        cls_out = x[:, 0, :]  # (B, embed_dim)
        return cls_out


class SubjectHead(nn.Module):
    """Single subject-specific classification/projection head."""

    def __init__(self, embed_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.head(x)


class EEGCNNTransformer(nn.Module):
    """
    Full architecture: CNN Feature Extractor → Transformer Backbone → Subject Heads.
    
    Supports both classification (output_dim=20) and retrieval (output_dim=CLIP embed_dim).
    """

    def __init__(
        self,
        # CNN params
        cnn_embed_dim: int = 64,
        cnn_num_layers: int = 2,
        cnn_kernel_size: int = 25,
        # Transformer params
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        feedforward_dim: int = 256,
        transformer_dropout: float = 0.1,
        # Head params
        head_hidden_dim: int = 128,
        output_dim: int = 20,  # 20 for classification, 512 for CLIP
        num_subjects: int = 13,
        head_dropout: float = 0.3,
        # General
        num_channels: int = 122,
    ):
        super().__init__()

        self.num_subjects = num_subjects
        self.output_dim = output_dim

        # Stage 1
        self.cnn = CNNFeatureExtractor(
            embed_dim=cnn_embed_dim,
            num_layers=cnn_num_layers,
            kernel_size=cnn_kernel_size,
            dropout=transformer_dropout,
        )

        # Stage 2
        self.backbone = TransformerBackbone(
            embed_dim=cnn_embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            feedforward_dim=feedforward_dim,
            dropout=transformer_dropout,
            num_electrodes=num_channels,
        )

        # Stage 3: Subject-specific heads
        self.heads = nn.ModuleList([
            SubjectHead(cnn_embed_dim, head_hidden_dim, output_dim, head_dropout)
            for _ in range(num_subjects)
        ])

    def forward(self, eeg: torch.Tensor, subject_idx: torch.Tensor):
        """
        Args:
            eeg:         (B, 122, 500)
            subject_idx: (B,) — integer subject IDs
        Returns:
            logits/embeddings: (B, output_dim)
        """
        # Stage 1: CNN feature extraction
        features = self.cnn(eeg)  # (B, 122, embed_dim)

        # Stage 2: Transformer backbone
        global_repr = self.backbone(features)  # (B, embed_dim)

        # Stage 3: Route to subject-specific heads (selective backprop)
        B = eeg.size(0)
        output = torch.zeros(B, self.output_dim, device=eeg.device)

        for sid in range(self.num_subjects):
            mask = subject_idx == sid
            if mask.any():
                output[mask] = self.heads[sid](global_repr[mask])

        return output

    def get_backbone_features(self, eeg: torch.Tensor):
        """Get shared backbone output (before subject heads) for analysis."""
        features = self.cnn(eeg)
        return self.backbone(features)

    def count_parameters(self):
        """Print parameter counts for each component."""
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.heads.parameters())
        total = sum(p.numel() for p in self.parameters())

        print(f"CNN Feature Extractor: {cnn_params:,} params")
        print(f"Transformer Backbone:  {backbone_params:,} params")
        print(f"Subject Heads (×{self.num_subjects}): {head_params:,} params")
        print(f"Total: {total:,} params")
        return total
