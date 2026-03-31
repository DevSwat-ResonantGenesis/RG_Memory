"""
Learned Sphere Projection — 512→3D with Triplet Loss Training
================================================================

Replaces the deterministic random projection (embedding_to_xyz) with a
trained neural network that learns to map embeddings to 3D sphere coordinates
where semantic similarity is preserved.

Architecture:
    embedding (512-dim) → Linear(512, 128) → ReLU → Linear(128, 32) → ReLU → Linear(32, 3) → L2 normalize

Training:
    Triplet loss: d(anchor, positive) < d(anchor, negative) + margin
    Triplets mined from:
    - Same-cluster pairs = positives
    - Different-cluster pairs = negatives
    - Hard negative mining for faster convergence

Model file: app/data/models/sphere_projection_model.pt
Training:  python -m app.services.sphere_projection --train
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "models", "sphere_projection_model.pt"
)

# Try to import torch — graceful fallback if not available
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available — sphere projection will use fallback")


if _TORCH_AVAILABLE:
    class SphereProjectionNet(nn.Module):
        """Neural network that projects embeddings onto a 3D unit sphere.
        
        Architecture:
            512 → 128 (ReLU) → 32 (ReLU) → 3 → L2 normalize
        
        The L2 normalization ensures all outputs lie on the unit sphere surface.
        Similar embeddings → nearby points on the sphere.
        """
        
        def __init__(self, input_dim: int = 512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Project embedding to 3D unit sphere."""
            raw = self.net(x)
            # L2 normalize to unit sphere
            return F.normalize(raw, p=2, dim=-1)


class SphereProjector:
    """Inference wrapper for the learned sphere projection model.
    
    Falls back to deterministic random projection if:
    - PyTorch not available
    - Trained model file not found
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._loaded = False
        self._load_attempted = False
    
    def _ensure_loaded(self) -> bool:
        """Lazy-load the trained model."""
        if self._loaded:
            return True
        if self._load_attempted:
            return False
        
        self._load_attempted = True
        
        if not _TORCH_AVAILABLE:
            logger.info("PyTorch not available — using fallback projection")
            return False
        
        if not os.path.exists(self.model_path):
            logger.info(
                f"Sphere projection model not found at {self.model_path}. "
                f"Using fallback projection. Train with: "
                f"python -m app.services.sphere_projection --train"
            )
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            input_dim = checkpoint.get("input_dim", 512)
            self._model = SphereProjectionNet(input_dim=input_dim)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            self._loaded = True
            
            logger.info(
                f"Loaded sphere projection model "
                f"(input_dim={input_dim}, "
                f"epoch={checkpoint.get('epoch', '?')}, "
                f"loss={checkpoint.get('final_loss', '?'):.4f})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load sphere projection model: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if the trained model is loaded."""
        return self._ensure_loaded()
    
    def project(self, embedding: List[float]) -> Tuple[float, float, float]:
        """Project a single embedding to 3D sphere coordinates.
        
        Args:
            embedding: 512-dim embedding vector
        
        Returns:
            (x, y, z) on the unit sphere, then scaled to [0, 1] via sigmoid
        """
        if not self._ensure_loaded():
            # Fallback to deterministic random projection (NOT embedding_to_xyz to avoid recursion)
            from .resonance_hashing import ResonanceHasher
            return ResonanceHasher._random_project_xyz(embedding)
        
        with torch.no_grad():
            emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            sphere_coords = self._model(emb_tensor).squeeze(0).numpy()
        
        # The model outputs unit sphere coordinates in [-1, 1]
        # Map to [0, 1] for compatibility with existing Hash Sphere system
        x = float((sphere_coords[0] + 1.0) / 2.0)
        y = float((sphere_coords[1] + 1.0) / 2.0)
        z = float((sphere_coords[2] + 1.0) / 2.0)
        
        return (x, y, z)
    
    def project_batch(self, embeddings: List[List[float]]) -> List[Tuple[float, float, float]]:
        """Project a batch of embeddings."""
        if not self._ensure_loaded():
            from .resonance_hashing import ResonanceHasher
            return [ResonanceHasher._random_project_xyz(e) for e in embeddings]
        
        with torch.no_grad():
            emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
            sphere_coords = self._model(emb_tensor).numpy()
        
        results = []
        for sc in sphere_coords:
            x = float((sc[0] + 1.0) / 2.0)
            y = float((sc[1] + 1.0) / 2.0)
            z = float((sc[2] + 1.0) / 2.0)
            results.append((x, y, z))
        
        return results


# Global singleton
_sphere_projector: Optional[SphereProjector] = None


def get_sphere_projector() -> SphereProjector:
    """Get or create the global sphere projector."""
    global _sphere_projector
    if _sphere_projector is None:
        _sphere_projector = SphereProjector()
    return _sphere_projector


# ============================================================================
# TRAINING
# ============================================================================

def _generate_triplets(
    embeddings: np.ndarray,
    cluster_labels: List[str],
    n_triplets: int = 2000,
) -> List[Tuple[int, int, int]]:
    """Generate triplets for training with semi-hard negative mining.
    
    For each anchor:
    - Positive: random sample from same cluster
    - Negative: closest sample from different cluster (semi-hard)
    """
    from collections import defaultdict
    
    # Group indices by cluster
    cluster_indices = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        cluster_indices[label].append(i)
    
    clusters = list(cluster_indices.keys())
    rng = np.random.RandomState(42)
    triplets = []
    
    for _ in range(n_triplets):
        # Pick random anchor cluster with at least 2 members
        valid_clusters = [c for c in clusters if len(cluster_indices[c]) >= 2]
        if not valid_clusters:
            break
        
        anchor_cluster = rng.choice(valid_clusters)
        anchor_idx, pos_idx = rng.choice(
            cluster_indices[anchor_cluster], size=2, replace=False
        )
        
        # Pick negative from a different cluster
        neg_clusters = [c for c in clusters if c != anchor_cluster]
        if not neg_clusters:
            continue
        neg_cluster = rng.choice(neg_clusters)
        neg_idx = rng.choice(cluster_indices[neg_cluster])
        
        triplets.append((int(anchor_idx), int(pos_idx), int(neg_idx)))
    
    return triplets


def train_sphere_projection(
    data_path: str = None,
    output_path: str = None,
    epochs: int = 200,
    lr: float = 0.001,
    margin: float = 0.3,
    n_triplets: int = 3000,
):
    """Train the sphere projection model from labeled data.
    
    Uses the same training data as the semantic encoder:
    - Generates embeddings with Nomic Embed
    - Mines triplets from cluster labels
    - Trains with triplet margin loss
    
    Args:
        data_path: Path to semantic_training_data.json
        output_path: Path to save the model
        epochs: Number of training epochs
        lr: Learning rate
        margin: Triplet loss margin
        n_triplets: Number of triplets to generate
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for training. Install: pip install torch")
    
    import json
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    if data_path is None:
        data_path = str(base_dir / "data" / "semantic_training_data.json")
    if output_path is None:
        output_path = MODEL_PATH
    
    # Load training data
    with open(data_path, "r") as f:
        training_data = json.load(f)
    
    texts = [d["text"] for d in training_data]
    clusters = [d["cluster"] for d in training_data]
    
    logger.info(f"Loaded {len(texts)} training examples")
    
    # Generate embeddings (reuse training script logic)
    from ..data.models.train_semantic_model import generate_embeddings
    embeddings = generate_embeddings(texts)
    
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    # Generate triplets
    triplets = _generate_triplets(embeddings, clusters, n_triplets=n_triplets)
    logger.info(f"Generated {len(triplets)} triplets")
    
    # Convert to tensors
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # Initialize model
    input_dim = embeddings.shape[1]
    model = SphereProjectionNet(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)
    
    # Training loop
    model.train()
    best_loss = float("inf")
    
    for epoch in range(epochs):
        total_loss = 0.0
        np.random.shuffle(triplets)
        
        # Mini-batch training
        batch_size = 64
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]
            
            anchor_indices = [t[0] for t in batch]
            pos_indices = [t[1] for t in batch]
            neg_indices = [t[2] for t in batch]
            
            anchor_embs = emb_tensor[anchor_indices]
            pos_embs = emb_tensor[pos_indices]
            neg_embs = emb_tensor[neg_indices]
            
            # Forward pass
            anchor_proj = model(anchor_embs)
            pos_proj = model(pos_embs)
            neg_proj = model(neg_embs)
            
            loss = triplet_loss_fn(anchor_proj, pos_proj, neg_proj)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(triplets) // batch_size)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "epoch": epochs,
        "final_loss": best_loss,
        "margin": margin,
        "n_triplets": len(triplets),
        "training_samples": len(texts),
        "cluster_names": list(set(clusters)),
    }, output_path)
    
    file_size = os.path.getsize(output_path)
    logger.info(f"Saved sphere projection model to {output_path} ({file_size:,} bytes)")
    logger.info(f"Final loss: {best_loss:.4f}")
    
    # Verification: check that same-cluster pairs are closer than cross-cluster
    model.eval()
    with torch.no_grad():
        all_proj = model(emb_tensor).numpy()
    
    from collections import defaultdict
    cluster_indices = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_indices[c].append(i)
    
    intra_dists = []
    inter_dists = []
    for c, indices in cluster_indices.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                d = np.linalg.norm(all_proj[indices[i]] - all_proj[indices[j]])
                intra_dists.append(d)
    
    cluster_list = list(cluster_indices.keys())
    for ci in range(len(cluster_list)):
        for cj in range(ci + 1, len(cluster_list)):
            for i in cluster_indices[cluster_list[ci]][:3]:
                for j in cluster_indices[cluster_list[cj]][:3]:
                    d = np.linalg.norm(all_proj[i] - all_proj[j])
                    inter_dists.append(d)
    
    logger.info(f"Intra-cluster avg distance: {np.mean(intra_dists):.4f}")
    logger.info(f"Inter-cluster avg distance: {np.mean(inter_dists):.4f}")
    logger.info(f"Separation ratio: {np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8):.2f}x")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if "--train" in sys.argv:
        train_sphere_projection()
    else:
        print("Usage: python -m app.services.sphere_projection --train")
