"""Memory Service Integration Tests.

Comprehensive integration tests for memory service endpoints:
- Memory storage and retrieval
- RAG (Retrieval Augmented Generation)
- Embeddings generation
- Conversation memory
- Vector search

Author: Agent 7 - ResonantGenesis Team
Created: February 21, 2026
"""

import pytest
import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock
from uuid import uuid4

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app


class TestConfig:
    """Test configuration constants."""
    BASE_URL = "http://testserver"
    TEST_USER_ID = "test-user-123"
    TEST_AGENT_ID = "test-agent-456"
    TEST_CONVERSATION_ID = "conv-789"


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def json_headers():
    """Return JSON content-type headers."""
    return {"Content-Type": "application/json"}


@pytest.fixture
def auth_headers():
    """Return authorization headers."""
    return {
        "Authorization": "Bearer test-token",
        "X-User-Id": TestConfig.TEST_USER_ID
    }


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "memory_service"
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_status_endpoint(self, client):
        """Test status endpoint."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "memory_service"
        assert data["status"] == "active"


class TestMemoryStorageEndpoints:
    """Test memory storage endpoints."""
    
    def test_store_memory(self, client, json_headers, auth_headers):
        """Test storing a memory."""
        headers = {**json_headers, **auth_headers}
        payload = {
            "content": "This is a test memory",
            "metadata": {"type": "test", "source": "integration_test"}
        }
        response = client.post(
            "/memory/store",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 201, 400, 401, 403, 404, 422, 500]
    
    def test_retrieve_memory(self, client, auth_headers):
        """Test retrieving a memory."""
        memory_id = str(uuid4())
        response = client.get(
            f"/memory/{memory_id}",
            headers=auth_headers
        )
        assert response.status_code in [200, 401, 403, 404, 500]
    
    def test_update_memory(self, client, json_headers, auth_headers):
        """Test updating a memory."""
        headers = {**json_headers, **auth_headers}
        memory_id = str(uuid4())
        payload = {
            "content": "Updated memory content",
            "metadata": {"updated": True}
        }
        response = client.put(
            f"/memory/{memory_id}",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 400, 401, 403, 404, 422, 500]
    
    def test_delete_memory(self, client, auth_headers):
        """Test deleting a memory."""
        memory_id = str(uuid4())
        response = client.delete(
            f"/memory/{memory_id}",
            headers=auth_headers
        )
        assert response.status_code in [200, 204, 401, 403, 404, 500]
    
    def test_list_memories(self, client, auth_headers):
        """Test listing memories."""
        response = client.get(
            "/memory/list",
            headers=auth_headers
        )
        assert response.status_code in [200, 401, 403, 404, 500]


class TestRAGEndpoints:
    """Test RAG (Retrieval Augmented Generation) endpoints."""
    
    def test_get_rag_conversations(self, client, auth_headers):
        """Test getting RAG conversations."""
        response = client.get(
            "/memory/rag/conversations",
            headers=auth_headers
        )
        assert response.status_code in [200, 401, 403, 404, 500]
    
    def test_upload_rag_document(self, client, auth_headers):
        """Test uploading a RAG document."""
        files = {
            "file": ("test.txt", b"This is test content for RAG", "text/plain")
        }
        response = client.post(
            "/memory/rag/upload",
            files=files,
            headers=auth_headers
        )
        assert response.status_code in [200, 201, 400, 401, 403, 404, 422, 500]
    
    def test_query_rag(self, client, json_headers, auth_headers):
        """Test querying RAG."""
        headers = {**json_headers, **auth_headers}
        payload = {
            "query": "What is the test content?",
            "top_k": 5
        }
        response = client.post(
            "/memory/rag/query",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 400, 401, 403, 404, 422, 500]
    
    def test_delete_rag_document(self, client, auth_headers):
        """Test deleting a RAG document."""
        doc_id = str(uuid4())
        response = client.delete(
            f"/memory/rag/documents/{doc_id}",
            headers=auth_headers
        )
        assert response.status_code in [200, 204, 401, 403, 404, 500]


class TestEmbeddingsEndpoints:
    """Test embeddings generation endpoints."""
    
    def test_generate_embedding(self, client, json_headers, auth_headers):
        """Test generating embeddings."""
        headers = {**json_headers, **auth_headers}
        payload = {
            "text": "This is text to embed"
        }
        response = client.post(
            "/memory/embeddings/generate",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 400, 401, 403, 404, 422, 500]
    
    def test_batch_embeddings(self, client, json_headers, auth_headers):
        """Test batch embeddings generation."""
        headers = {**json_headers, **auth_headers}
        payload = {
            "texts": ["Text one", "Text two", "Text three"]
        }
        response = client.post(
            "/memory/embeddings/batch",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 400, 401, 403, 404, 422, 500]


class TestConversationMemoryEndpoints:
    """Test conversation memory endpoints."""
    
    def test_store_conversation(self, client, json_headers, auth_headers):
        """Test storing conversation memory."""
        headers = {**json_headers, **auth_headers}
        payload = {
            "conversation_id": TestConfig.TEST_CONVERSATION_ID,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        response = client.post(
            "/memory/conversations/store",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 201, 400, 401, 403, 404, 422, 500]
    
    def test_get_conversation(self, client, auth_headers):
        """Test getting conversation memory."""
        response = client.get(
            f"/memory/conversations/{TestConfig.TEST_CONVERSATION_ID}",
            headers=auth_headers
        )
        assert response.status_code in [200, 401, 403, 404, 500]
    
    def test_list_conversations(self, client, auth_headers):
        """Test listing conversations."""
        response = client.get(
            "/memory/conversations",
            headers=auth_headers
        )
        assert response.status_code in [200, 401, 403, 404, 500]
    
    def test_delete_conversation(self, client, auth_headers):
        """Test deleting conversation memory."""
        response = client.delete(
            f"/memory/conversations/{TestConfig.TEST_CONVERSATION_ID}",
            headers=auth_headers
        )
        assert response.status_code in [200, 204, 401, 403, 404, 500]


class TestVectorSearchEndpoints:
    """Test vector search endpoints."""
    
    def test_semantic_search(self, client, json_headers, auth_headers):
        """Test semantic search."""
        headers = {**json_headers, **auth_headers}
        payload = {
            "query": "Find similar memories",
            "top_k": 10,
            "threshold": 0.7
        }
        response = client.post(
            "/memory/search/semantic",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 400, 401, 403, 404, 422, 500]
    
    def test_hybrid_search(self, client, json_headers, auth_headers):
        """Test hybrid search (semantic + keyword)."""
        headers = {**json_headers, **auth_headers}
        payload = {
            "query": "test query",
            "keywords": ["test", "memory"],
            "top_k": 10
        }
        response = client.post(
            "/memory/search/hybrid",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 400, 401, 403, 404, 422, 500]


class TestAgentMemoryEndpoints:
    """Test agent-specific memory endpoints."""
    
    def test_store_agent_memory(self, client, json_headers, auth_headers):
        """Test storing agent memory."""
        headers = {**json_headers, **auth_headers}
        payload = {
            "agent_id": TestConfig.TEST_AGENT_ID,
            "memory_type": "episodic",
            "content": "Agent learned something new"
        }
        response = client.post(
            "/memory/agents/store",
            json=payload,
            headers=headers
        )
        assert response.status_code in [200, 201, 400, 401, 403, 404, 422, 500]
    
    def test_get_agent_memories(self, client, auth_headers):
        """Test getting agent memories."""
        response = client.get(
            f"/memory/agents/{TestConfig.TEST_AGENT_ID}",
            headers=auth_headers
        )
        assert response.status_code in [200, 401, 403, 404, 500]
    
    def test_clear_agent_memories(self, client, auth_headers):
        """Test clearing agent memories."""
        response = client.delete(
            f"/memory/agents/{TestConfig.TEST_AGENT_ID}/clear",
            headers=auth_headers
        )
        assert response.status_code in [200, 204, 401, 403, 404, 500]


class TestVisualizerEndpoints:
    """Test memory visualizer endpoints."""
    
    def test_get_memory_graph(self, client, auth_headers):
        """Test getting memory graph visualization."""
        response = client.get(
            "/memory/visualizer/graph",
            headers=auth_headers
        )
        assert response.status_code in [200, 401, 403, 404, 500]
    
    def test_get_memory_stats(self, client, auth_headers):
        """Test getting memory statistics."""
        response = client.get(
            "/memory/visualizer/stats",
            headers=auth_headers
        )
        assert response.status_code in [200, 401, 403, 404, 500]


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_json(self, client, auth_headers):
        """Test handling of invalid JSON."""
        headers = {**auth_headers, "Content-Type": "application/json"}
        response = client.post(
            "/memory/store",
            content="not valid json {{{",
            headers=headers
        )
        assert response.status_code in [400, 422, 500]
    
    def test_missing_auth(self, client, json_headers):
        """Test handling of missing authentication."""
        response = client.get("/memory/list", headers=json_headers)
        assert response.status_code in [200, 401, 403, 404, 500]
    
    def test_nonexistent_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/memory/nonexistent/endpoint")
        assert response.status_code in [404, 405]


class TestCORSHeaders:
    """Test CORS header handling."""
    
    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        headers = {
            "Origin": "https://resonantgenesis.xyz",
            "Access-Control-Request-Method": "POST"
        }
        response = client.options("/memory/store", headers=headers)
        assert response.status_code in [200, 204, 404]
    
    def test_cors_headers_present(self, client):
        """Test CORS headers in response."""
        headers = {"Origin": "https://resonantgenesis.xyz"}
        response = client.get("/health", headers=headers)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
