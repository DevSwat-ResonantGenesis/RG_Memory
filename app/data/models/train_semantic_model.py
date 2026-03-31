"""
Train Semantic Encoder Model for Hash Sphere Memory System
============================================================

Trains cluster classifier + temperature/polarity regressors from labeled data.
Uses the same Nomic Embed model that production uses for embedding generation.

Input:  semantic_training_data.json (50+ labeled examples)
Output: trained_semantic_encoder.pkl (sklearn models + cluster centroids)

Usage:
    python -m app.data.models.train_semantic_model
    
    OR from project root:
    python app/data/models/train_semantic_model.py
"""

import json
import os
import pickle
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cluster mapping (V0.1 Spec: Alpha/Beta/Gamma + Delta/Epsilon/Zeta)
CLUSTER_NAMES = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA"]


def load_training_data(data_path: str) -> List[Dict]:
    """Load labeled training data from JSON."""
    with open(data_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} training examples from {data_path}")
    return data


def generate_embeddings(texts: List[str], model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> np.ndarray:
    """Generate embeddings using the same model as production."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Add task prefix (same as production NomicEmbeddings.encode)
    prefixed = [f"search_document: {t}" for t in texts]
    
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(prefixed, convert_to_tensor=True)
    
    # Apply Matryoshka reduction to 512-dim (same as production)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :512]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()


def train_cluster_classifier(
    embeddings: np.ndarray,
    cluster_labels: List[str],
) -> Tuple[KNeighborsClassifier, LabelEncoder, Dict[str, np.ndarray]]:
    """Train KNN cluster classifier from embeddings.
    
    Returns:
        - Trained KNN classifier
        - Label encoder (string → int mapping)
        - Cluster centroids {cluster_name: mean_embedding}
    """
    le = LabelEncoder()
    y = le.fit_transform(cluster_labels)
    
    # KNN with k=5 (good for small datasets, captures local structure)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(embeddings) - 1), weights="distance")
    knn.fit(embeddings, y)
    
    # Cross-validation accuracy
    if len(embeddings) >= 10:
        scores = cross_val_score(knn, embeddings, y, cv=min(5, len(embeddings)))
        logger.info(f"Cluster classifier CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    # Compute cluster centroids (mean embedding per cluster)
    centroids = {}
    for cluster_name in le.classes_:
        mask = np.array(cluster_labels) == cluster_name
        if mask.sum() > 0:
            centroids[cluster_name] = embeddings[mask].mean(axis=0)
    
    logger.info(f"Trained cluster classifier with {len(le.classes_)} clusters: {list(le.classes_)}")
    return knn, le, centroids


def train_temperature_regressor(
    embeddings: np.ndarray,
    temperatures: List[float],
) -> Ridge:
    """Train temperature regressor (embedding → temperature 0-1)."""
    reg = Ridge(alpha=1.0)
    reg.fit(embeddings, temperatures)
    
    # R² score
    r2 = reg.score(embeddings, temperatures)
    logger.info(f"Temperature regressor R²: {r2:.3f}")
    return reg


def train_polarity_regressor(
    embeddings: np.ndarray,
    polarities: List[float],
) -> Ridge:
    """Train polarity regressor (embedding → polarity 0-1)."""
    reg = Ridge(alpha=1.0)
    reg.fit(embeddings, polarities)
    
    r2 = reg.score(embeddings, polarities)
    logger.info(f"Polarity regressor R²: {r2:.3f}")
    return reg


def train_and_save(data_path: str = None, output_path: str = None):
    """Full training pipeline: load data → embed → train → save."""
    # Resolve paths
    base_dir = Path(__file__).parent
    if data_path is None:
        data_path = str(base_dir.parent / "semantic_training_data.json")
        if not os.path.exists(data_path):
            data_path = str(base_dir / ".." / "semantic_training_data.json")
    if output_path is None:
        output_path = str(base_dir / "trained_semantic_encoder.pkl")
    
    # Load training data
    training_data = load_training_data(data_path)
    
    texts = [d["text"] for d in training_data]
    clusters = [d["cluster"] for d in training_data]
    temperatures = [d["temperature"] for d in training_data]
    polarities = [d["polarity"] for d in training_data]
    
    # Generate embeddings
    embeddings = generate_embeddings(texts)
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    # Train models
    cluster_clf, label_encoder, centroids = train_cluster_classifier(embeddings, clusters)
    temp_reg = train_temperature_regressor(embeddings, temperatures)
    polarity_reg = train_polarity_regressor(embeddings, polarities)
    
    # Package everything
    model_package = {
        "version": "2.0",
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        "embedding_dim": embeddings.shape[1],
        "matryoshka_dim": 512,
        "cluster_classifier": cluster_clf,
        "label_encoder": label_encoder,
        "cluster_centroids": centroids,
        "temperature_regressor": temp_reg,
        "polarity_regressor": polarity_reg,
        "training_samples": len(training_data),
        "cluster_names": list(label_encoder.classes_),
    }
    
    # Save
    with open(output_path, "wb") as f:
        pickle.dump(model_package, f)
    
    file_size = os.path.getsize(output_path)
    logger.info(f"Saved trained model to {output_path} ({file_size:,} bytes)")
    logger.info(f"Clusters: {model_package['cluster_names']}")
    logger.info(f"Training samples: {model_package['training_samples']}")
    
    # Verify by loading and predicting
    logger.info("\n--- Verification ---")
    with open(output_path, "rb") as f:
        loaded = pickle.load(f)
    
    for text, emb, true_cluster, true_temp, true_pol in zip(
        texts[:5], embeddings[:5], clusters[:5], temperatures[:5], polarities[:5]
    ):
        pred_cluster_idx = loaded["cluster_classifier"].predict(emb.reshape(1, -1))[0]
        pred_cluster = loaded["label_encoder"].inverse_transform([pred_cluster_idx])[0]
        pred_temp = float(np.clip(loaded["temperature_regressor"].predict(emb.reshape(1, -1))[0], 0, 1))
        pred_pol = float(np.clip(loaded["polarity_regressor"].predict(emb.reshape(1, -1))[0], 0, 1))
        
        logger.info(
            f'  "{text[:50]}..." → '
            f'cluster={pred_cluster} (true={true_cluster}) | '
            f'temp={pred_temp:.2f} (true={true_temp:.2f}) | '
            f'pol={pred_pol:.2f} (true={true_pol:.2f})'
        )
    
    return model_package


if __name__ == "__main__":
    train_and_save()
