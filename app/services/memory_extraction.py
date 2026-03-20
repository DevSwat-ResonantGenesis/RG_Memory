"""
Memory Extraction Service
Implements multi-method memory retrieval for Hash Sphere:
- Anchor-based lookup (fast)
- Semantic proximity search (accurate)
- Resonance-based filtering (quality)
- Cluster-based retrieval (context)
- Multi-method ranking (combined scoring)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from sqlmodel import Session, select

from ..models.governance.resonant_chat import MemoryAnchor, ResonanceCluster, ResonantChatMessage, ResonantChat
from .resonance_hashing import ResonanceHasher
from .hybrid_memory_ranker import rank_memories


class MemoryExtractionService:
    """Extract relevant memories using multiple methods"""
    
    def __init__(self):
        self.hasher = ResonanceHasher()
        # PATCH #47: Neural Gravity Engine
        try:
            from .neural_gravity_engine import NeuralGravityEngine
            self.gravity_engine = NeuralGravityEngine()
        except ImportError:
            self.gravity_engine = None
    
    def magnetic_pull(self, resonance_score: float) -> float:
        """
        Patch #39: Hash Sphere Magnetic Pull System (HS-MPS)
        
        Non-linear boost to strong memories. Weak memories stay weak,
        strong memories get amplified, creating a "magnetic field" effect
        that pulls responses toward core stable meaning.
        
        Args:
            resonance_score: Original resonance score (0.0 to 1.0)
        
        Returns:
            Magnetic-pulled score (amplified for high resonance)
        """
        # Non-linear boost: square the score and multiply by 1.5
        # This means:
        # - Low resonance (0.3) -> 0.3^2 * 1.5 = 0.135 (weaker)
        # - Medium resonance (0.6) -> 0.6^2 * 1.5 = 0.54 (moderate)
        # - High resonance (0.9) -> 0.9^2 * 1.5 = 1.215 (strong, but capped at 1.0)
        magnetic = (resonance_score ** 2) * 1.5
        
        # Cap at 1.0 to prevent overflow
        return min(magnetic, 1.0)
    
    def extract_memories(
        self,
        session: Session,
        user_id: UUID,
        org_id: UUID,
        query: str,
        query_hash: str,
        query_xyz: Tuple[float, float, float],
        limit: int = 5,
        use_anchors: bool = True,
        use_proximity: bool = True,
        use_resonance: bool = True,
        use_clusters: bool = True,
        agent_hash: Optional[str] = None,  # Shared agent support
    ) -> List[Dict]:
        """
        Extract relevant memories using multi-method approach.
        
        Args:
            session: Database session
            user_id: User ID
            org_id: Organization ID
            query: User query text
            query_hash: Hash of query
            query_xyz: XYZ coordinates of query
            limit: Maximum number of memories to return
            use_anchors: Enable anchor-based lookup
            use_proximity: Enable semantic proximity search
            use_resonance: Enable resonance-based filtering
            use_clusters: Enable cluster-based retrieval
        
        Returns:
            List of memory dictionaries with scores
        """
        all_memories: Dict[str, Dict] = {}
        
        # 1. Anchor-based lookup (fast)
        if use_anchors:
            anchor_memories = self._extract_by_anchors(session, user_id, org_id, query, agent_hash=agent_hash)
            for mem in anchor_memories:
                mem_id = mem.get("id")
                if mem_id:
                    if mem_id not in all_memories:
                        all_memories[mem_id] = mem
                    # Add anchor score
                    all_memories[mem_id]["anchor_score"] = mem.get("anchor_score", 0.0)
        
        # 2. Semantic proximity search (accurate)
        if use_proximity:
            proximity_memories = self._extract_by_proximity(session, user_id, org_id, query_xyz, agent_hash=agent_hash)
            for mem in proximity_memories:
                mem_id = mem.get("id")
                if mem_id:
                    if mem_id not in all_memories:
                        all_memories[mem_id] = mem
                    # Add proximity score
                    all_memories[mem_id]["proximity_score"] = mem.get("proximity_score", 0.0)
        
        # 3. Resonance-based filtering (quality)
        if use_resonance:
            resonance_memories = self._extract_by_resonance(session, user_id, org_id, query_hash, agent_hash=agent_hash)
            for mem in resonance_memories:
                mem_id = mem.get("id")
                if mem_id:
                    if mem_id not in all_memories:
                        all_memories[mem_id] = mem
                    # Add resonance score
                    all_memories[mem_id]["resonance_score"] = mem.get("resonance_score", 0.0)
        
        # 4. Cluster-based retrieval (context)
        if use_clusters:
            cluster_memories = self._extract_by_clusters(session, user_id, org_id, query_xyz, agent_hash=agent_hash)
            for mem in cluster_memories:
                mem_id = mem.get("id")
                if mem_id:
                    if mem_id not in all_memories:
                        all_memories[mem_id] = mem
                    # Add cluster score as RAG score (PATCH #11)
                    cluster_score = mem.get("cluster_score", 0.0)
                    all_memories[mem_id]["cluster_score"] = cluster_score
                    # Map cluster_score to rag_score for hybrid ranking
                    all_memories[mem_id]["rag_score"] = cluster_score
        
        # PATCH #11: Combine results from all methods and prepare for hybrid ranking
        combined = list(all_memories.values())
        
        # ============================================
        # LAYER 5: SPHERE GEOMETRY - Resonance Function
        # R(h) = sin(a·x) + cos(b·y) + tan(c·z)
        # ============================================
        # Calculate resonance function for query point
        query_resonance_value = self.hasher.calculate_resonance_function(query_xyz)
        
        # ============================================
        # LAYER 4: ANCHOR RESONANCE - Anchor Energy
        # E_j(s) = exp(-β·||s - A_j||²)
        # ============================================
        # Collect anchor positions for energy calculation
        anchor_positions = []
        for mem in combined:
            if mem.get("xyz") and mem.get("anchor_score", 0) > 0.3:
                anchor_positions.append(np.array(mem["xyz"]))
        
        # Ensure all memories have required score fields for hybrid ranking
        for mem in combined:
            # Map existing scores to hybrid ranker fields
            if "rag_score" not in mem:
                # Use cluster_score or similarity_score as rag_score
                mem["rag_score"] = mem.get("cluster_score") or mem.get("similarity_score") or 0.0
            if "resonance_score" not in mem:
                mem["resonance_score"] = 0.0
            if "proximity_score" not in mem:
                # Calculate if we have XYZ coordinates
                if mem.get("xyz") and query_xyz:
                    mem_xyz = mem["xyz"]
                    if mem_xyz and all(x is not None for x in mem_xyz):
                        mem["proximity_score"] = self.hasher.calculate_proximity_score(query_xyz, mem_xyz)
                    else:
                        mem["proximity_score"] = 0.0
                else:
                    mem["proximity_score"] = 0.0
            if "recency_score" not in mem:
                # Calculate recency score
                recency_score = 0.0
                if mem.get("timestamp"):
                    try:
                        timestamp = datetime.fromisoformat(mem["timestamp"].replace('Z', '+00:00'))
                        age_days = (datetime.now(timestamp.tzinfo) - timestamp).days
                        recency_score = np.exp(-age_days / 30.0)  # Half-life of 30 days
                    except Exception:
                        recency_score = 0.5  # Default if parsing fails
                mem["recency_score"] = recency_score
            if "anchor_score" not in mem:
                mem["anchor_score"] = mem.get("importance_score", 0.0)
            
            # ============================================
            # LAYER 5: Calculate resonance function score for memory
            # ============================================
            if mem.get("xyz") and all(x is not None for x in mem["xyz"]):
                mem_resonance_value = self.hasher.calculate_resonance_function(mem["xyz"])
                # Similarity between query and memory resonance (normalized)
                # Resonance function returns values roughly in [-3, 3]
                resonance_diff = abs(query_resonance_value - mem_resonance_value)
                mem["resonance_function_score"] = max(0.0, 1.0 - (resonance_diff / 6.0))
            else:
                mem["resonance_function_score"] = 0.0
            
            # ============================================
            # LAYER 4: Calculate anchor energy for memory
            # ============================================
            if mem.get("xyz") and anchor_positions:
                query_point = np.array(query_xyz)
                mem_point = np.array(mem["xyz"])
                # Find best anchor energy for this memory point
                _, best_energy = self.hasher.find_best_anchor(mem_point, anchor_positions, beta=1.0)
                mem["anchor_energy"] = best_energy
            else:
                mem["anchor_energy"] = 0.0
        
        # PATCH #11: Apply hybrid ranking using the new ranker
        ranked_memories = rank_memories(combined)
        
        # Return top-k memories
        return ranked_memories[:limit]
    
    def _extract_by_anchors(
        self,
        session: Session,
        user_id: UUID,
        org_id: UUID,
        query: str,
        agent_hash: Optional[str] = None,
    ) -> List[Dict]:
        """Extract memories by anchor-based lookup (fast keyword matching)"""
        # Extract keywords from query
        keywords = self.hasher.extract_anchors(query, max_anchors=10)
        
        if not keywords:
            return []
        
        # Find anchors matching keywords
        # If agent_hash provided, query by agent_hash (shared memory)
        # Otherwise, query by user_id (backward compatible)
        if agent_hash:
            anchor_query = select(MemoryAnchor).where(
                MemoryAnchor.agent_hash == agent_hash,
            )
        else:
            anchor_query = select(MemoryAnchor).where(
                MemoryAnchor.user_id == user_id,
                MemoryAnchor.org_id == org_id,
            )
        
        # Match any of the keywords (OR condition)
        # Simple approach: check if anchor_text contains any keyword
        matching_anchors = []
        for anchor in session.exec(anchor_query).all():
            anchor_lower = anchor.anchor_text.lower()
            for keyword in keywords:
                if keyword.lower() in anchor_lower:
                    matching_anchors.append(anchor)
                    break
        
        # Get messages/contexts from matching anchors
        # SECURITY: Verify message belongs to user via chat
        # If agent_hash provided, query chats by agent_hash (shared memory)
        # Otherwise, query by user_id (backward compatible)
        if agent_hash:
            user_chats = session.exec(
                select(ResonantChat.id).where(
                    ResonantChat.agent_hash == agent_hash
                )
            ).all()
        else:
            user_chats = session.exec(
                select(ResonantChat.id).where(
                    ResonantChat.user_id == user_id,
                    ResonantChat.org_id == org_id
                )
            ).all()
        
        chat_ids_list = list(user_chats) if user_chats else []
        
        memories = []
        for anchor in matching_anchors:
            if anchor.message_id:
                # Get message and verify it belongs to user's chat (SECURITY: Only user's own messages)
                if chat_ids_list:
                    msg_query = select(ResonantChatMessage).where(
                        ResonantChatMessage.id == anchor.message_id,
                        ResonantChatMessage.chat_id.in_(chat_ids_list)
                    )
                else:
                    # No chats = no messages
                    continue
                msg = session.exec(msg_query).first()
                if msg:
                    memories.append({
                        "id": str(msg.id),
                        "type": "message",
                        "content": msg.content,
                        "hash": msg.hash,
                        "xyz": (msg.xyz_x, msg.xyz_y, msg.xyz_z) if msg.xyz_x is not None else None,
                        "resonance_score": msg.resonance_score or 0.0,
                        "anchor_score": anchor.importance_score,
                        "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                    })
            elif anchor.context:
                # Use anchor context
                memories.append({
                    "id": str(anchor.id),
                    "type": "anchor",
                    "content": anchor.context,
                    "hash": anchor.anchor_hash,
                    "xyz": (anchor.xyz_x, anchor.xyz_y, anchor.xyz_z) if anchor.xyz_x is not None else None,
                    "resonance_score": 0.0,
                    "anchor_score": anchor.importance_score,
                    "timestamp": anchor.created_at.isoformat() if anchor.created_at else None,
                })
        
        return memories
    
    def _extract_by_proximity(
        self,
        session: Session,
        user_id: UUID,
        org_id: UUID,
        query_xyz: Tuple[float, float, float],
        limit: int = 10,
        agent_hash: Optional[str] = None,
    ) -> List[Dict]:
        """Extract memories by semantic proximity (3D distance)"""
        # SECURITY: Filter by user_id and org_id via chat, or by agent_hash for shared memory
        # First, get all chats for this user/org or agent_hash
        if agent_hash:
            user_chats = session.exec(
                select(ResonantChat.id).where(
                    ResonantChat.agent_hash == agent_hash
                )
            ).all()
        else:
            user_chats = session.exec(
                select(ResonantChat.id).where(
                    ResonantChat.user_id == user_id,
                    ResonantChat.org_id == org_id
                )
            ).all()
        
        if not user_chats:
            return []
        
        # Get messages from user's chats only (SECURITY: Only user's own messages)
        chat_ids_list = list(user_chats)
        msg_query = select(ResonantChatMessage).where(
            ResonantChatMessage.chat_id.in_(chat_ids_list),
            ResonantChatMessage.hash.isnot(None),
            ResonantChatMessage.xyz_x.isnot(None),
        )
        
        messages = session.exec(msg_query).all()
        
        # Calculate distances and sort
        memories_with_distance = []
        for msg in messages:
            if msg.xyz_x is not None and msg.xyz_y is not None and msg.xyz_z is not None:
                msg_xyz = (msg.xyz_x, msg.xyz_y, msg.xyz_z)
                distance = self.hasher.calculate_proximity(query_xyz, msg_xyz)
                proximity_score = self.hasher.calculate_proximity_score(query_xyz, msg_xyz)
                
                memories_with_distance.append({
                    "id": str(msg.id),
                    "type": "message",
                    "content": msg.content,
                    "hash": msg.hash,
                    "xyz": msg_xyz,
                    "resonance_score": msg.resonance_score or 0.0,
                    "proximity_score": proximity_score,
                    "distance": distance,
                    "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                })
        
        # Sort by distance (closer = better)
        memories_with_distance.sort(key=lambda x: x["distance"])
        
        return memories_with_distance[:limit]
    
    def _extract_by_resonance(
        self,
        session: Session,
        user_id: UUID,
        org_id: UUID,
        query_hash: str,
        limit: int = 10,
        agent_hash: Optional[str] = None,
    ) -> List[Dict]:
        """Extract memories by resonance score (hash similarity)"""
        # SECURITY: Filter by user_id and org_id via chat, or by agent_hash for shared memory
        # First, get all chats for this user/org or agent_hash
        if agent_hash:
            user_chats = session.exec(
                select(ResonantChat.id).where(
                    ResonantChat.agent_hash == agent_hash
                )
            ).all()
        else:
            user_chats = session.exec(
                select(ResonantChat.id).where(
                    ResonantChat.user_id == user_id,
                    ResonantChat.org_id == org_id
                )
            ).all()
        
        if not user_chats:
            return []
        
        # Get messages from user's chats only (SECURITY: Only user's own messages)
        chat_ids_list = list(user_chats)
        msg_query = select(ResonantChatMessage).where(
            ResonantChatMessage.chat_id.in_(chat_ids_list),
            ResonantChatMessage.hash.isnot(None),
        )
        
        messages = session.exec(msg_query).all()
        
        # Calculate resonance scores
        memories_with_resonance = []
        for msg in messages:
            if msg.hash:
                resonance = self.hasher.calculate_resonance(query_hash, msg.hash)
                
                memories_with_resonance.append({
                    "id": str(msg.id),
                    "type": "message",
                    "content": msg.content,
                    "hash": msg.hash,
                    "xyz": (msg.xyz_x, msg.xyz_y, msg.xyz_z) if msg.xyz_x is not None else None,
                    "resonance_score": resonance,
                    "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                })
        
        # Sort by resonance (higher = better)
        memories_with_resonance.sort(key=lambda x: x["resonance_score"], reverse=True)
        
        return memories_with_resonance[:limit]
    
    def _extract_by_clusters(
        self,
        session: Session,
        user_id: UUID,
        org_id: UUID,
        query_xyz: Tuple[float, float, float],
        limit: int = 5,
        agent_hash: Optional[str] = None,
    ) -> List[Dict]:
        """Extract memories by cluster (context-based)"""
        # Get all clusters
        cluster_query = select(ResonanceCluster).where(
            ResonanceCluster.user_id == user_id,
            ResonanceCluster.org_id == org_id,
        )
        clusters = session.exec(cluster_query).all()
        
        # Find cluster closest to query XYZ
        closest_cluster = None
        min_distance = float('inf')
        
        for cluster in clusters:
            # Use cluster hash to approximate XYZ (if available)
            # For now, use cluster resonance score as proxy
            # In production, store cluster XYZ coordinates
            distance = 1.0 - cluster.resonance_score  # Simple proxy
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster
        
        if not closest_cluster:
            return []
        
        # Get memories from cluster anchors
        # SECURITY: Verify messages belong to user via chat, or by agent_hash for shared memory
        if agent_hash:
            user_chats = session.exec(
                select(ResonantChat.id).where(
                    ResonantChat.agent_hash == agent_hash
                )
            ).all()
        else:
            user_chats = session.exec(
                select(ResonantChat.id).where(
                    ResonantChat.user_id == user_id,
                    ResonantChat.org_id == org_id
                )
            ).all()
        
        chat_ids_list = list(user_chats) if user_chats else []
        
        memories = []
        for anchor_id in closest_cluster.anchor_ids:
            anchor_query = select(MemoryAnchor).where(MemoryAnchor.id == UUID(anchor_id))
            anchor = session.exec(anchor_query).first()
            if anchor and anchor.message_id:
                # Verify message belongs to user's chat (SECURITY: Only user's own messages)
                if chat_ids_list:
                    msg_query = select(ResonantChatMessage).where(
                        ResonantChatMessage.id == anchor.message_id,
                        ResonantChatMessage.chat_id.in_(chat_ids_list)
                    )
                else:
                    # No chats = no messages
                    continue
                msg = session.exec(msg_query).first()
                if msg:
                    memories.append({
                        "id": str(msg.id),
                        "type": "message",
                        "content": msg.content,
                        "hash": msg.hash,
                        "xyz": (msg.xyz_x, msg.xyz_y, msg.xyz_z) if msg.xyz_x is not None else None,
                        "resonance_score": msg.resonance_score or 0.0,
                        "cluster_score": closest_cluster.resonance_score,
                        "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                    })
        
        return memories[:limit]
    
    def _rank_memories(
        self,
        memories: List[Dict],
        query_xyz: Tuple[float, float, float],
    ) -> List[Dict]:
        """
        Rank memories using multi-method scoring.
        
        Scoring weights:
        - Resonance: 0.4 (semantic similarity)
        - Proximity: 0.3 (3D distance)
        - Anchor: 0.2 (keyword match)
        - Recency: 0.1 (time-based)
        """
        for mem in memories:
            # Get individual scores (default to 0 if not present)
            resonance_score = mem.get("resonance_score", 0.0)
            proximity_score = mem.get("proximity_score", 0.0)
            anchor_score = mem.get("anchor_score", 0.0)
            
            # Calculate recency score (0-1, newer = higher)
            recency_score = 0.0
            if mem.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(mem["timestamp"].replace('Z', '+00:00'))
                    age_days = (datetime.now(timestamp.tzinfo) - timestamp).days
                    # Exponential decay: newer = higher score
                    recency_score = np.exp(-age_days / 30.0)  # Half-life of 30 days
                except Exception:
                    recency_score = 0.5  # Default if parsing fails
            
            # If proximity not calculated, calculate it now
            if proximity_score == 0.0 and mem.get("xyz"):
                mem_xyz = mem["xyz"]
                if mem_xyz and all(x is not None for x in mem_xyz):
                    proximity_score = self.hasher.calculate_proximity_score(query_xyz, mem_xyz)
                    mem["proximity_score"] = proximity_score
            
            # ============================================
            # PATCH #39: Hash Sphere Magnetic Pull System (HS-MPS)
            # ============================================
            # Non-linear boost to strong memories - creates magnetic field effect
            magnetic = self.magnetic_pull(resonance_score)
            
            # ============================================
            # PATCH #47: Hash Sphere Neural Gravity Engine (NG-Engine)
            # ============================================
            # Apply gravitational forces from high-resonance anchors
            gravity_force = 0.0
            if self.gravity_engine and mem.get("xyz") and query_xyz:
                # Get top anchors for gravity calculation (from all memories)
                # For now, use current memory as anchor (simplified)
                # In full implementation, would use all high-importance anchors
                if mem.get("anchor_score", 0.0) > 0.5:  # Only high-importance anchors
                    anchor_xyz = mem.get("xyz")
                    if anchor_xyz:
                        gravity_force = self.gravity_engine.compute_gravity(
                            query_xyz,
                            anchor_xyz,
                            strength=mem.get("anchor_score", 1.0)
                        )
            
            # Combined score with magnetic pull and gravity
            combined_score = (
                magnetic * 0.4 +  # Magnetic pull weight
                proximity_score * 0.25 +
                anchor_score * 0.15 +
                recency_score * 0.10 +
                gravity_force * 0.10  # PATCH #47: Gravity force
            )
            
            mem["combined_score"] = combined_score
            mem["magnetic_score"] = magnetic  # Store magnetic score for debugging
            mem["gravity_force"] = gravity_force  # Store gravity force for debugging
        
        # Sort by combined score (higher = better)
        memories.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        return memories

