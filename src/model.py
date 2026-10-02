"""
model.py
Phase 4: Transformer-based Sequence Classifier for Indian Sign Language (ISL).

Architecture:
1. Linear Input Projection: projects raw landmark input_dim (e.g. 1662) -> d_model (256).
2. Input Dropout: regularizes projection before sequence modeling.
3. Positional Encoding: sinusoidal temporal positional encoding to retain temporal order.
4. Transformer Encoder: 3-layer TransformerEncoder (d_model=256, nhead=8, dim_feedforward=512, dropout=0.3, batch_first=True).
5. Temporal Aggregation: sequence pooling (mean-pooling or attention-weighted pooling) across T=30 frames.
6. Classification Head: Linear(256, 128) -> LayerNorm(128) -> ReLU -> Dropout(0.3) -> Linear(128, num_classes).
"""

import os
import sys
import math

# Auto-fallback to isl_py310 conda environment if torch is not in system python
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    conda_py = r"C:\Users\abhis\anaconda3\anaconda\envs\isl_py310\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        result = subprocess.run([conda_py] + sys.argv)
        sys.exit(result.returncode)
    raise


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding to inject temporal sequence order.
    Matches batch_first=True format: (batch_size, seq_len, d_model).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """
    Self-attention weighted temporal pooling across the sequence dimension.
    Learns to dynamically weight key action frames over static or transitional frames.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional boolean mask (batch_size, seq_len) where True indicates padding.
        Returns:
            Pooled tensor of shape (batch_size, d_model).
        """
        # Calculate attention scores: (batch_size, seq_len, 1)
        scores = self.attn_net(x)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(x * weights, dim=1)  # (batch_size, d_model)
        return pooled


class ISLTransformerClassifier(nn.Module):
    """
    Transformer-based sequence classifier for Indian Sign Language recognition.

    Args:
        input_dim: Number of input features per frame (e.g., 1662 for holistic landmarks).
        num_classes: Number of gloss target classes (e.g., 37).
        d_model: Hidden dimensionality of the transformer (default: 256).
        nhead: Number of multi-head attention heads (default: 8).
        dim_feedforward: Hidden dimension of the feedforward network (default: 512).
        num_layers: Number of TransformerEncoder layers (default: 3).
        dropout: Dropout rate across layers (default: 0.3).
        pooling: Temporal aggregation method: 'mean' or 'attention' (default: 'mean').
        max_seq_len: Maximum supported sequence length for positional encoding (default: 100).
    """

    def __init__(
        self,
        input_dim: int = 226,
        num_classes: int = 37,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 2,
        dropout: float = 0.35,
        pooling: str = "mean",
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.pooling_type = pooling.lower()

        # 1. Linear Input Projection & Normalization
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_seq_len)

        # 3. Transformer Encoder (Pre-LN for gradient stability on landmark sequences)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # 4. Temporal Aggregation Layer
        if self.pooling_type == "attention":
            self.temporal_pooling = AttentionPooling(d_model=d_model)
        elif self.pooling_type == "mean":
            self.temporal_pooling = None  # Handled via x.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}. Choose 'mean' or 'attention'.")

        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier uniform initialization for stable convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the ISL Transformer Classifier.

        Args:
            x: Input tensor of shape (batch_size, seq_len=30, input_dim=1662)
            key_padding_mask: Optional boolean mask (batch_size, seq_len) for padding frames.

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # 1. Project input features to d_model and normalize
        x = self.input_projection(x)  # (B, T, d_model)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # 2. Add Positional Encoding
        x = self.pos_encoder(x)  # (B, T, d_model)

        # 3. Transformer Encoder Processing
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)  # (B, T, d_model)

        # 4. Temporal Aggregation across sequence length (dimension 1)
        if self.pooling_type == "attention":
            x = self.temporal_pooling(x, mask=key_padding_mask)  # (B, d_model)
        else:
            x = x.mean(dim=1)  # (B, d_model)

        # 5. Classification Head
        logits = self.classifier(x)  # (B, num_classes)

        return logits


if __name__ == "__main__":
    print("=== Testing ISL Transformer Classifier Architecture ===")

    # Model Configuration
    BATCH_SIZE = 8
    SEQ_LEN = 30
    FEATURE_DIM = 226  # Upper pose (100) + Left Hand (63) + Right Hand (63)
    NUM_CLASSES = 37

    # Initialize model with regularized architecture
    model = ISLTransformerClassifier(
        input_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        num_layers=2,
        dropout=0.35,
        pooling="mean",
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Input Shape:          ({BATCH_SIZE}, {SEQ_LEN}, {FEATURE_DIM})")
    print(f"Target Classes:       {NUM_CLASSES}")

    # Generate dummy input tensor
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM)

    # Test in training mode
    model.train()
    out_train = model(dummy_input)
    print(f"\n[Train Mode] Forward Pass Output Shape: {out_train.shape}")
    assert out_train.shape == (BATCH_SIZE, NUM_CLASSES), f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {out_train.shape}"
    assert not torch.isnan(out_train).any(), "NaN found in training output logits!"

    # Test in evaluation mode
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_input)
    print(f"[Eval Mode]  Forward Pass Output Shape: {out_eval.shape}")
    assert out_eval.shape == (BATCH_SIZE, NUM_CLASSES), f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {out_eval.shape}"
    assert not torch.isnan(out_eval).any(), "NaN found in evaluation output logits!"

    # Test attention pooling mode
    attn_model = ISLTransformerClassifier(
        input_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        pooling="attention",
    )
    attn_out = attn_model(dummy_input)
    assert attn_out.shape == (BATCH_SIZE, NUM_CLASSES), "Attention pooling shape check failed!"
    print(f"[Attention Pooling] Output Shape:       {attn_out.shape}")

    print("\n=== All Architecture & Shape Checks Passed Successfully! ===")
