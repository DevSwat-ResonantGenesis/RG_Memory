"""
Trained HashSphere Encoder — Semantic XYZ from Neural Hashes
=============================================================

Loads the trained EncoderNetwork (4.2MB) from RG_HashSphere_Memory.
Converts MiniLM 384-dim embeddings → 160-bit semantic hashes → deterministic XYZ.

Benefits over SHA-256 + trig:
  - Similar text → nearby 160-bit hashes → nearby XYZ coordinates
  - Trained via contrastive learning, not random
  - Deterministic: same embedding always → same hash → same XYZ

Falls back to random projection if model file is unavailable.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Model file location — can be set via env var or volume mount
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "universe_encoder.pt"

_encoder = None
_available = False


def _load_encoder():
    """Load the trained EncoderNetwork weights (lazy, once)."""
    global _encoder, _available

    if _encoder is not None:
        return

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.warning("PyTorch not available — trained hash encoder disabled")
        _available = False
        return

    if not MODEL_PATH.exists():
        logger.info(f"Trained encoder not found at {MODEL_PATH} — using fallback")
        _available = False
        return

    class EncoderNetwork(nn.Module):
        def __init__(self, embedding_dim: int = 384, hash_bits: int = 160):
            super().__init__()
            self.hash_bits = hash_bits
            self.encoder = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, hash_bits),
                nn.Sigmoid(),
            )

        def forward(self, embedding):
            soft_bits = self.encoder(embedding)
            hard_bits = (soft_bits > 0.5).float()
            return hard_bits

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        net = EncoderNetwork()
        if "encoder_state_dict" in checkpoint:
            net.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
        elif "encoder" in checkpoint:
            net.load_state_dict(checkpoint["encoder"], strict=False)
        else:
            net.load_state_dict(checkpoint, strict=False)
        net.eval()
        _encoder = net
        _available = True
        logger.info(f"✅ Trained HashSphere encoder loaded ({MODEL_PATH.stat().st_size / 1e6:.1f}MB)")
    except Exception as e:
        logger.warning(f"Failed to load trained encoder: {e}")
        _available = False


def embedding_to_hash(embedding: List[float]) -> Optional[str]:
    """
    Convert a 384-dim MiniLM embedding to a 160-bit hex hash using the trained encoder.

    Returns:
        40-char hex string, or None if encoder unavailable
    """
    _load_encoder()
    if not _available or _encoder is None:
        return None

    try:
        import torch
        with torch.no_grad():
            emb_tensor = torch.tensor([embedding], dtype=torch.float32)
            hard_bits = _encoder(emb_tensor)[0]  # shape: (160,)
            bits = hard_bits.numpy().astype(np.uint8)
            # Pack 160 bits into 20 bytes → 40 hex chars
            byte_array = np.packbits(bits)
            return byte_array.tobytes().hex()
    except Exception as e:
        logger.warning(f"Trained encoder inference failed: {e}")
        return None


def hash_to_xyz(hex_hash: str) -> Tuple[float, float, float]:
    """
    Convert a 40-char hex hash to deterministic XYZ in [0, 1] range.

    Splits 160 bits into 3 groups of ~53 bits, normalizes each to [0, 1].
    """
    try:
        raw = bytes.fromhex(hex_hash)
        # Use first 8 bytes for X, next 8 for Y, last 4 for Z
        x_bytes = raw[0:8]
        y_bytes = raw[8:16]
        z_bytes = raw[16:20]

        x = struct.unpack(">Q", x_bytes)[0] / (2**64 - 1)
        y = struct.unpack(">Q", y_bytes)[0] / (2**64 - 1)
        z = struct.unpack(">I", z_bytes)[0] / (2**32 - 1)

        return (x, y, z)
    except Exception:
        return (0.5, 0.5, 0.5)


def embedding_to_xyz(embedding: List[float]) -> Optional[Tuple[float, float, float]]:
    """
    Full pipeline: embedding → trained hash → XYZ.
    Returns None if trained encoder is unavailable (caller should fallback).
    """
    hex_hash = embedding_to_hash(embedding)
    if hex_hash is None:
        return None
    return hash_to_xyz(hex_hash)


def is_available() -> bool:
    """Check if the trained encoder is loaded and ready."""
    _load_encoder()
    return _available
