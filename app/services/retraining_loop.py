"""
Autonomous Retraining Loop for Hash Sphere ML Models
======================================================

Background service that periodically retrains:
1. Semantic encoder (cluster classifier + temperature/polarity regressors)
2. Sphere projection (triplet-loss 512→3D neural network)

Data sources:
- Production MemoryEmbedding table (embeddings already computed at ingest)
- Cluster labels derived from existing anchors and content analysis
- User feedback signals (future: explicit relevance ratings)

Triggers:
- Time-based: every N hours (configurable)
- Count-based: every N new memories ingested
- Manual: via API endpoint

Safety:
- Models are saved to temp files first, then atomically renamed
- Old models are backed up before replacement
- Training runs in a background thread to not block the event loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Configuration
RETRAIN_INTERVAL_HOURS = 24  # Retrain every 24 hours
RETRAIN_MIN_SAMPLES = 100    # Minimum memories before retraining is worthwhile
RETRAIN_NEW_THRESHOLD = 50   # Retrain when 50+ new memories since last train

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
SEMANTIC_MODEL_PATH = MODELS_DIR / "trained_semantic_encoder.pkl"
SPHERE_MODEL_PATH = MODELS_DIR / "sphere_projection_model.pt"
RETRAIN_STATE_PATH = MODELS_DIR / "retrain_state.json"


class RetrainingState:
    """Persisted state tracking when retraining last ran and what changed."""
    
    def __init__(self, state_path: str = str(RETRAIN_STATE_PATH)):
        self.state_path = state_path
        self._state = self._load()
    
    def _load(self) -> Dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "last_retrain_timestamp": None,
            "last_retrain_sample_count": 0,
            "total_retrains": 0,
            "last_semantic_loss": None,
            "last_sphere_loss": None,
            "memories_at_last_train": 0,
        }
    
    def save(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self._state, f, indent=2, default=str)
    
    @property
    def last_retrain(self) -> Optional[datetime]:
        ts = self._state.get("last_retrain_timestamp")
        if ts:
            return datetime.fromisoformat(ts)
        return None
    
    @property
    def memories_at_last_train(self) -> int:
        return self._state.get("memories_at_last_train", 0)
    
    def record_retrain(self, sample_count: int, semantic_loss: float = None, sphere_loss: float = None):
        self._state["last_retrain_timestamp"] = datetime.utcnow().isoformat()
        self._state["last_retrain_sample_count"] = sample_count
        self._state["total_retrains"] = self._state.get("total_retrains", 0) + 1
        self._state["memories_at_last_train"] = sample_count
        if semantic_loss is not None:
            self._state["last_semantic_loss"] = semantic_loss
        if sphere_loss is not None:
            self._state["last_sphere_loss"] = sphere_loss
        self.save()


class AutonomousRetrainer:
    """Background retraining loop for Hash Sphere ML models.
    
    Checks periodically if retraining is needed based on:
    1. Time elapsed since last retrain
    2. Number of new memories since last retrain
    
    When triggered, runs training in a background thread.
    """
    
    def __init__(
        self,
        interval_hours: float = RETRAIN_INTERVAL_HOURS,
        min_samples: int = RETRAIN_MIN_SAMPLES,
        new_threshold: int = RETRAIN_NEW_THRESHOLD,
    ):
        self.interval_hours = interval_hours
        self.min_samples = min_samples
        self.new_threshold = new_threshold
        self.state = RetrainingState()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._training_in_progress = False
    
    async def start(self):
        """Start the autonomous retraining loop."""
        if self._running:
            logger.warning("Retraining loop already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Autonomous retraining loop started "
            f"(interval={self.interval_hours}h, "
            f"min_samples={self.min_samples}, "
            f"new_threshold={self.new_threshold})"
        )
    
    async def stop(self):
        """Stop the retraining loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Autonomous retraining loop stopped")
    
    async def _loop(self):
        """Main loop — check conditions and trigger retrain."""
        while self._running:
            try:
                # Check every 10 minutes
                await asyncio.sleep(600)
                
                if self._training_in_progress:
                    continue
                
                should_retrain, reason = await self._should_retrain()
                if should_retrain:
                    logger.info(f"Retraining triggered: {reason}")
                    await self._run_retrain()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retraining loop error: {e}")
                await asyncio.sleep(60)
    
    async def _should_retrain(self) -> Tuple[bool, str]:
        """Check if retraining should be triggered."""
        # Get current memory count
        current_count = await self._get_memory_count()
        
        if current_count < self.min_samples:
            return False, f"Not enough samples ({current_count} < {self.min_samples})"
        
        # Time-based trigger
        if self.state.last_retrain:
            hours_since = (datetime.utcnow() - self.state.last_retrain).total_seconds() / 3600
            if hours_since >= self.interval_hours:
                return True, f"Time trigger: {hours_since:.1f}h since last retrain"
        else:
            # Never trained before
            return True, "First-time training"
        
        # Count-based trigger
        new_memories = current_count - self.state.memories_at_last_train
        if new_memories >= self.new_threshold:
            return True, f"Count trigger: {new_memories} new memories"
        
        return False, "No trigger met"
    
    async def _get_memory_count(self) -> int:
        """Get current count of memories with embeddings using existing async pool."""
        try:
            from ..db import get_session
            from sqlalchemy import text
            
            async for session in get_session():
                result = await session.execute(
                    text("SELECT COUNT(*) FROM memory_embeddings WHERE embedding IS NOT NULL")
                )
                count = result.scalar() or 0
                return count
        except Exception as e:
            logger.warning(f"Failed to get memory count: {e}")
            return 0
    
    async def _run_retrain(self):
        """Export data async (existing pool), then train sync in thread (no DB)."""
        self._training_in_progress = True
        
        try:
            # Step 1: Export training data ASYNC — uses existing session pool, no second pool
            training_data = await self._export_training_data_async()
            if not training_data or len(training_data) < self.min_samples:
                logger.info(f"Not enough exportable data: {len(training_data) if training_data else 0}")
                return
            
            logger.info(f"Exported {len(training_data)} samples, starting training in background thread")
            
            # Step 2: Train in thread — pure compute, ZERO DB access
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._retrain_sync, training_data)
            
            if result:
                sample_count, semantic_loss, sphere_loss = result
                self.state.record_retrain(
                    sample_count=sample_count,
                    semantic_loss=semantic_loss,
                    sphere_loss=sphere_loss,
                )
                logger.info(
                    f"Retraining complete: {sample_count} samples, "
                    f"semantic_loss={semantic_loss}, sphere_loss={sphere_loss}"
                )
                
                # Reload models in memory
                self._reload_models()
            else:
                logger.warning("Retraining returned no result")
                
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
        finally:
            self._training_in_progress = False
    
    def _retrain_sync(self, training_data: List[Dict]) -> Optional[Tuple[int, Optional[float], Optional[float]]]:
        """Synchronous training logic (runs in thread). NO DB ACCESS.
        
        Args:
            training_data: Pre-exported list of dicts with embedding/cluster/temperature/polarity
        
        Returns:
            (sample_count, semantic_loss, sphere_loss) or None on failure
        """
        try:
            sample_count = len(training_data)
            semantic_loss = None
            sphere_loss = None
            
            # Train semantic encoder (pure compute)
            try:
                semantic_loss = self._retrain_semantic_encoder(training_data)
            except Exception as e:
                logger.error(f"Semantic encoder retrain failed: {e}")
            
            # Train sphere projection (pure compute)
            try:
                sphere_loss = self._retrain_sphere_projection(training_data)
            except Exception as e:
                logger.error(f"Sphere projection retrain failed: {e}")
            
            return (sample_count, semantic_loss, sphere_loss)
            
        except Exception as e:
            logger.error(f"Retraining sync failed: {e}")
            return None
    
    async def _export_training_data_async(self) -> Optional[List[Dict]]:
        """Export training data using the EXISTING async session pool.
        
        No second connection pool — reuses the same asyncpg pool as the rest
        of the app. Data is loaded into memory here, then passed to the
        sync training thread which does zero DB access.
        """
        try:
            from ..db import get_session
            from sqlalchemy import text
            
            training_data = []
            
            async for session in get_session():
                result = await session.execute(text("""
                    SELECT 
                        e.embedding,
                        m.content,
                        m.intensity_score,
                        m.sentiment_score,
                        m.cluster_name,
                        m.extra_metadata
                    FROM memory_embeddings e
                    JOIN memory_records m ON e.memory_id = m.id
                    WHERE e.embedding IS NOT NULL
                    AND m.content IS NOT NULL
                    AND length(m.content) > 10
                    LIMIT 5000
                """))
                rows = result.fetchall()
            
            for row in rows:
                embedding = row[0]
                content = row[1]
                intensity = float(row[2]) if row[2] is not None else 0.5
                sentiment = float(row[3]) if row[3] is not None else 0.5
                cluster_name = row[4]
                extra_metadata = row[5]
                
                if embedding is None or not content:
                    continue
                
                # Derive cluster label
                cluster = "GAMMA"  # Default
                if cluster_name:
                    name_upper = cluster_name.upper()
                    for c in ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA"]:
                        if c in name_upper:
                            cluster = c
                            break
                elif extra_metadata and isinstance(extra_metadata, dict):
                    cluster = extra_metadata.get("cluster", "GAMMA")
                else:
                    cluster = self._derive_cluster_label_from_text(content)
                
                training_data.append({
                    "embedding": list(embedding) if not isinstance(embedding, list) else embedding,
                    "cluster": cluster,
                    "temperature": intensity,
                    "polarity": sentiment,
                    "text": content[:200],
                })
            
            logger.info(f"Exported {len(training_data)} training samples from memory DB (async pool)")
            return training_data
                
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _derive_cluster_label_from_text(self, text: str) -> str:
        """Derive cluster label from text content using hardcoded semantic encoder."""
        try:
            from .semantic_encoder import get_semantic_encoder
            enc = get_semantic_encoder()
            result = enc.encode(text[:2000])
            return result.dominant_cluster.name
        except Exception:
            return "GAMMA"
    
    def _retrain_semantic_encoder(self, training_data: List[Dict]) -> Optional[float]:
        """Retrain the semantic encoder from production data."""
        import pickle
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import LabelEncoder
        
        embeddings = np.array([d["embedding"] for d in training_data])
        clusters = [d["cluster"] for d in training_data]
        temperatures = [d["temperature"] for d in training_data]
        polarities = [d["polarity"] for d in training_data]
        
        # Train cluster classifier
        le = LabelEncoder()
        y = le.fit_transform(clusters)
        knn = KNeighborsClassifier(
            n_neighbors=min(7, len(embeddings) - 1),
            weights="distance"
        )
        knn.fit(embeddings, y)
        
        # Compute centroids
        centroids = {}
        for name in le.classes_:
            mask = np.array(clusters) == name
            if mask.sum() > 0:
                centroids[name] = embeddings[mask].mean(axis=0)
        
        # Train regressors
        temp_reg = Ridge(alpha=1.0)
        temp_reg.fit(embeddings, temperatures)
        
        pol_reg = Ridge(alpha=1.0)
        pol_reg.fit(embeddings, polarities)
        
        model_package = {
            "version": "2.0-auto",
            "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
            "embedding_dim": embeddings.shape[1],
            "matryoshka_dim": 512,
            "cluster_classifier": knn,
            "label_encoder": le,
            "cluster_centroids": centroids,
            "temperature_regressor": temp_reg,
            "polarity_regressor": pol_reg,
            "training_samples": len(training_data),
            "cluster_names": list(le.classes_),
            "retrained_at": datetime.utcnow().isoformat(),
        }
        
        # Atomic save: write to temp, then rename
        temp_path = str(SEMANTIC_MODEL_PATH) + ".tmp"
        backup_path = str(SEMANTIC_MODEL_PATH) + ".bak"
        
        with open(temp_path, "wb") as f:
            pickle.dump(model_package, f)
        
        # Backup old model
        if os.path.exists(str(SEMANTIC_MODEL_PATH)):
            shutil.copy2(str(SEMANTIC_MODEL_PATH), backup_path)
        
        # Atomic rename
        os.rename(temp_path, str(SEMANTIC_MODEL_PATH))
        
        r2 = temp_reg.score(embeddings, temperatures)
        logger.info(f"Semantic encoder retrained: {len(training_data)} samples, temp R²={r2:.3f}")
        return r2
    
    def _retrain_sphere_projection(self, training_data: List[Dict]) -> Optional[float]:
        """Retrain the sphere projection model from production data."""
        try:
            import torch
            import torch.nn as nn
            from .sphere_projection import SphereProjectionNet, _generate_triplets
        except ImportError:
            logger.warning("PyTorch not available — skipping sphere projection retrain")
            return None
        
        embeddings = np.array([d["embedding"] for d in training_data])
        clusters = [d["cluster"] for d in training_data]
        
        # Generate triplets
        triplets = _generate_triplets(embeddings, clusters, n_triplets=min(5000, len(embeddings) * 50))
        if len(triplets) < 100:
            logger.warning(f"Not enough triplets ({len(triplets)}) — skipping sphere projection")
            return None
        
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
        input_dim = embeddings.shape[1]
        
        model = SphereProjectionNet(input_dim=input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        triplet_loss_fn = nn.TripletMarginLoss(margin=0.3)
        
        model.train()
        best_loss = float("inf")
        
        for epoch in range(150):
            np.random.shuffle(triplets)
            total_loss = 0.0
            batch_size = 64
            
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                anchor_embs = emb_tensor[[t[0] for t in batch]]
                pos_embs = emb_tensor[[t[1] for t in batch]]
                neg_embs = emb_tensor[[t[2] for t in batch]]
                
                anchor_proj = model(anchor_embs)
                pos_proj = model(pos_embs)
                neg_proj = model(neg_embs)
                
                loss = triplet_loss_fn(anchor_proj, pos_proj, neg_proj)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(triplets) // batch_size)
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        # Atomic save
        temp_path = str(SPHERE_MODEL_PATH) + ".tmp"
        backup_path = str(SPHERE_MODEL_PATH) + ".bak"
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "epoch": 150,
            "final_loss": best_loss,
            "margin": 0.3,
            "n_triplets": len(triplets),
            "training_samples": len(training_data),
            "cluster_names": list(set(clusters)),
            "retrained_at": datetime.utcnow().isoformat(),
        }, temp_path)
        
        if os.path.exists(str(SPHERE_MODEL_PATH)):
            shutil.copy2(str(SPHERE_MODEL_PATH), backup_path)
        
        os.rename(temp_path, str(SPHERE_MODEL_PATH))
        
        logger.info(f"Sphere projection retrained: {len(training_data)} samples, loss={best_loss:.4f}")
        return best_loss
    
    def _reload_models(self):
        """Force reload of models in memory after retraining."""
        try:
            from .trained_semantic_encoder import get_trained_encoder
            enc = get_trained_encoder()
            enc._loaded = False
            enc._load_attempted = False
            enc._ensure_loaded()
        except Exception as e:
            logger.warning(f"Failed to reload semantic encoder: {e}")
        
        try:
            from .sphere_projection import get_sphere_projector
            proj = get_sphere_projector()
            proj._loaded = False
            proj._load_attempted = False
            proj._ensure_loaded()
        except Exception as e:
            logger.warning(f"Failed to reload sphere projector: {e}")
    
    async def trigger_manual_retrain(self) -> Dict:
        """Manually trigger a retrain. Returns status dict."""
        if self._training_in_progress:
            return {"status": "already_running", "message": "Training is already in progress"}
        
        logger.info("Manual retrain triggered")
        await self._run_retrain()
        
        return {
            "status": "completed",
            "last_retrain": self.state.last_retrain.isoformat() if self.state.last_retrain else None,
            "total_retrains": self.state._state.get("total_retrains", 0),
            "sample_count": self.state._state.get("last_retrain_sample_count", 0),
        }


# Global singleton
_retrainer: Optional[AutonomousRetrainer] = None


def get_retrainer() -> AutonomousRetrainer:
    """Get or create the global retrainer."""
    global _retrainer
    if _retrainer is None:
        _retrainer = AutonomousRetrainer()
    return _retrainer
