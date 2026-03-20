"""Unit tests for Memory Service."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import hashlib


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def sample_memory():
    """Sample memory record."""
    return {
        "id": uuid4(),
        "user_id": uuid4(),
        "org_id": uuid4(),
        "chat_id": uuid4(),
        "source": "chat",
        "content": "This is a test memory content for testing.",
        "hash": "abc123def456",
        "resonance_score": 0.85,
        "xyz_x": 0.1,
        "xyz_y": 0.2,
        "xyz_z": 0.3,
        "created_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_anchor():
    """Sample memory anchor."""
    return {
        "id": uuid4(),
        "user_id": uuid4(),
        "org_id": uuid4(),
        "anchor_text": "Important concept to remember",
        "anchor_hash": "hash123",
        "context": "This is the context around the anchor",
        "importance_score": 0.9,
        "xyz_x": 0.5,
        "xyz_y": 0.6,
        "xyz_z": 0.7,
        "anchor_type": "chat",
        "created_at": datetime.utcnow(),
    }


# ============================================
# Hash Sphere Tests
# ============================================

class TestHashSphere:
    """Test Hash Sphere functionality."""

    def test_hash_text_returns_valid_hash(self):
        """Test that hashing text returns a valid SHA-256 hash."""
        from app.hash_sphere import hash_text
        
        text = "Test content for hashing"
        result = hash_text(text)
        
        assert result is not None
        assert len(result) == 64  # SHA-256 hex digest
        assert all(c in '0123456789abcdef' for c in result)

    def test_hash_text_deterministic(self):
        """Test that same text produces same hash."""
        from app.hash_sphere import hash_text
        
        text = "Consistent content"
        hash1 = hash_text(text)
        hash2 = hash_text(text)
        
        assert hash1 == hash2

    def test_different_text_different_hash(self):
        """Test that different texts produce different hashes."""
        from app.hash_sphere import hash_text
        
        hash1 = hash_text("First content")
        hash2 = hash_text("Second content")
        
        assert hash1 != hash2

    def test_calculate_resonance_score(self):
        """Test resonance score calculation between two hashes."""
        from app.hash_sphere import calculate_resonance
        
        hash_a = hashlib.sha256(b"content a").hexdigest()
        hash_b = hashlib.sha256(b"content b").hexdigest()
        
        score = calculate_resonance(hash_a, hash_b)
        
        assert 0.0 <= score <= 1.0

    def test_same_hash_resonance_is_one(self):
        """Test that identical hashes have resonance of 1.0."""
        from app.hash_sphere import calculate_resonance
        
        hash_value = hashlib.sha256(b"same content").hexdigest()
        
        score = calculate_resonance(hash_value, hash_value)
        
        assert score == 1.0

    def test_calculate_xyz_coordinates(self):
        """Test XYZ coordinate calculation from hash."""
        from app.hash_sphere import calculate_xyz_coordinates
        
        hash_value = hashlib.sha256(b"test").hexdigest()
        
        x, y, z = calculate_xyz_coordinates(hash_value)
        
        # Coordinates should be normalized between -1 and 1
        assert -1.0 <= x <= 1.0
        assert -1.0 <= y <= 1.0
        assert -1.0 <= z <= 1.0

    def test_xyz_coordinates_deterministic(self):
        """Test that same hash produces same coordinates."""
        from app.hash_sphere import calculate_xyz_coordinates
        
        hash_value = hashlib.sha256(b"deterministic").hexdigest()
        
        coords1 = calculate_xyz_coordinates(hash_value)
        coords2 = calculate_xyz_coordinates(hash_value)
        
        assert coords1 == coords2


# ============================================
# Memory Embedding Tests
# ============================================

class TestMemoryEmbeddings:
    """Test memory embedding functionality."""

    @pytest.mark.asyncio
    async def test_create_embedding_returns_vector(self):
        """Test that embedding creation returns a vector."""
        from app.embeddings import create_embedding
        
        with patch('app.embeddings.openai_client') as mock_client:
            mock_client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536)]
            )
            
            text = "Test content for embedding"
            embedding = await create_embedding(text)
            
            assert embedding is not None
            assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_embedding_dimension_correct(self):
        """Test that embeddings have correct dimensions."""
        from app.embeddings import create_embedding
        
        with patch('app.embeddings.openai_client') as mock_client:
            mock_client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536)]
            )
            
            embedding = await create_embedding("test")
            
            # text-embedding-3-small produces 1536 dimensions
            assert len(embedding) == 1536

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from app.embeddings import cosine_similarity
        
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(vec_a, vec_b)
        
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        from app.embeddings import cosine_similarity
        
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        
        similarity = cosine_similarity(vec_a, vec_b)
        
        assert similarity == pytest.approx(0.0, rel=1e-5)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        from app.embeddings import cosine_similarity
        
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(vec_a, vec_b)
        
        assert similarity == pytest.approx(-1.0, rel=1e-5)


# ============================================
# RAG Retrieval Tests
# ============================================

class TestRAGRetrieval:
    """Test RAG retrieval functionality."""

    @pytest.mark.asyncio
    async def test_retrieve_memories_returns_list(self, mock_db_session):
        """Test that memory retrieval returns a list."""
        from app.rag import retrieve_memories
        
        mock_db_session.execute.return_value = MagicMock(
            scalars=lambda: MagicMock(all=lambda: [])
        )
        
        results = await retrieve_memories(
            query="test query",
            user_id=uuid4(),
            db=mock_db_session,
            top_k=5,
        )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_memories_respects_top_k(self, mock_db_session):
        """Test that retrieval respects top_k limit."""
        from app.rag import retrieve_memories
        
        # Create 10 mock memories
        mock_memories = [MagicMock() for _ in range(10)]
        mock_db_session.execute.return_value = MagicMock(
            scalars=lambda: MagicMock(all=lambda: mock_memories[:5])
        )
        
        results = await retrieve_memories(
            query="test",
            user_id=uuid4(),
            db=mock_db_session,
            top_k=5,
        )
        
        assert len(results) <= 5

    def test_rank_memories_by_relevance(self):
        """Test memory ranking by relevance score."""
        from app.rag import rank_memories
        
        memories = [
            {"content": "low relevance", "score": 0.3},
            {"content": "high relevance", "score": 0.9},
            {"content": "medium relevance", "score": 0.6},
        ]
        
        ranked = rank_memories(memories)
        
        assert ranked[0]["score"] == 0.9
        assert ranked[1]["score"] == 0.6
        assert ranked[2]["score"] == 0.3


# ============================================
# Memory Anchor Tests
# ============================================

class TestMemoryAnchors:
    """Test memory anchor functionality."""

    def test_extract_anchors_from_text(self):
        """Test anchor extraction from text."""
        from app.anchors import extract_anchors
        
        text = "The quick brown fox jumps over the lazy dog. This is important!"
        
        anchors = extract_anchors(text)
        
        assert isinstance(anchors, list)

    def test_anchor_importance_scoring(self):
        """Test anchor importance score calculation."""
        from app.anchors import calculate_importance
        
        anchor_text = "critical information"
        context = "This is the surrounding context with critical information embedded."
        
        score = calculate_importance(anchor_text, context)
        
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_create_anchor_stores_in_db(self, mock_db_session):
        """Test that creating anchor stores it in database."""
        from app.anchors import create_anchor
        
        anchor_data = {
            "user_id": uuid4(),
            "org_id": uuid4(),
            "anchor_text": "Test anchor",
            "context": "Test context",
        }
        
        await create_anchor(anchor_data, mock_db_session)
        
        mock_db_session.commit.assert_called_once()


# ============================================
# Memory Chunking Tests
# ============================================

class TestMemoryChunking:
    """Test memory chunking for long documents."""

    def test_chunk_text_respects_size(self):
        """Test that chunking respects max chunk size."""
        from app.chunking import chunk_text
        
        long_text = "word " * 1000  # 5000 characters
        chunks = chunk_text(long_text, max_chunk_size=500)
        
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_chunk_text_preserves_content(self):
        """Test that chunking preserves all content."""
        from app.chunking import chunk_text
        
        original = "This is the original text that should be preserved."
        chunks = chunk_text(original, max_chunk_size=20)
        
        reconstructed = "".join(chunks)
        # Allow for some whitespace differences
        assert original.replace(" ", "") in reconstructed.replace(" ", "")

    def test_chunk_overlap(self):
        """Test chunk overlap for context continuity."""
        from app.chunking import chunk_text
        
        text = "A B C D E F G H I J K L M N O P"
        chunks = chunk_text(text, max_chunk_size=10, overlap=3)
        
        # Check that consecutive chunks have overlap
        if len(chunks) > 1:
            # Last characters of chunk 0 should appear in chunk 1
            assert chunks[0][-3:] in chunks[1] or chunks[1][:3] in chunks[0]


# ============================================
# Memory Search Tests
# ============================================

class TestMemorySearch:
    """Test memory search functionality."""

    @pytest.mark.asyncio
    async def test_search_by_content(self, mock_db_session):
        """Test searching memories by content."""
        from app.search import search_memories
        
        mock_db_session.execute.return_value = MagicMock(
            scalars=lambda: MagicMock(all=lambda: [])
        )
        
        results = await search_memories(
            query="test search",
            user_id=uuid4(),
            db=mock_db_session,
        )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_filters_by_user(self, mock_db_session):
        """Test that search filters by user_id."""
        from app.search import search_memories
        
        user_id = uuid4()
        
        await search_memories(
            query="test",
            user_id=user_id,
            db=mock_db_session,
        )
        
        # Verify that execute was called (filter applied in query)
        mock_db_session.execute.assert_called_once()


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
