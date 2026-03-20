import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from starlette.requests import Request


def _make_request(headers: dict) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/memory/retrieve",
        "headers": [(k.lower().encode("utf-8"), str(v).encode("utf-8")) for k, v in headers.items()],
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_phase2_retrieve_filters_premium_agent_global_when_not_entitled():
    from app.routers import retrieve_memory, MemoryRetrieveRequest
    from app.services.pgvector_search import VectorSearchResult

    user_id = str(uuid4())
    org_id = str(uuid4())
    agent_hash = "agent_abc"

    payload = MemoryRetrieveRequest(
        user_id=user_id,
        org_id=org_id,
        agent_hash=agent_hash,
        query="hello",
        limit=10,
        use_vector_search=True,
    )

    request = _make_request({"x-is-dev-override": "false"})
    session = AsyncMock()

    overlay = VectorSearchResult(
        memory_id="m_overlay",
        content="overlay",
        similarity=0.9,
        metadata={"record_user_id": user_id, "record_agent_hash": agent_hash},
    )
    user_global = VectorSearchResult(
        memory_id="m_user_global",
        content="user_global",
        similarity=0.8,
        metadata={"record_user_id": user_id, "record_agent_hash": None},
    )
    agent_public = VectorSearchResult(
        memory_id="m_agent_public",
        content="agent_public",
        similarity=0.7,
        metadata={"tier": "public", "record_user_id": None, "record_agent_hash": agent_hash},
    )
    agent_premium = VectorSearchResult(
        memory_id="m_agent_premium",
        content="agent_premium",
        similarity=0.6,
        metadata={"tier": "premium", "record_user_id": None, "record_agent_hash": agent_hash},
    )

    async def _search_similar_side_effect(*, user_id=None, org_id=None, agent_hash=None, require_agent_hash_null=False, require_user_id_null=False, **kwargs):
        if user_id is not None and agent_hash is not None:
            return [overlay]
        if user_id is not None and require_agent_hash_null:
            return [user_global]
        if require_user_id_null and agent_hash is not None:
            return [agent_public, agent_premium]
        return []

    class _Resp:
        status_code = 200

        def json(self):
            return {"allowed": False}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _Resp()

    with (
        patch("app.routers.semantic_cache.get", return_value=None),
        patch("app.routers.semantic_cache.set", return_value=None),
        patch("app.routers.embedding_cache.get", return_value=None),
        patch("app.routers.embedding_cache.set", return_value=None),
        patch("app.routers.embeddings_generator.generate", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]])),
        patch("app.routers.pgvector_search.check_pgvector_available", new=AsyncMock(return_value=True)),
        patch("app.routers.pgvector_search.search_similar", new=AsyncMock(side_effect=_search_similar_side_effect)),
        patch("httpx.AsyncClient", _Client),
    ):
        results = await retrieve_memory(payload=payload, request=request, session=session)

    assert any(r.content == "overlay" for r in results)
    assert any(r.content == "user_global" for r in results)
    assert any(r.content == "agent_public" for r in results)
    assert not any(r.content == "agent_premium" for r in results)

    by_content = {r.content: r for r in results}
    assert by_content["overlay"].scope == "user_overlay"
    assert by_content["overlay"].tier == "private"
    assert by_content["user_global"].scope == "user_global"
    assert by_content["user_global"].tier == "private"
    assert by_content["agent_public"].scope == "agent_global"
    assert by_content["agent_public"].tier == "public"


@pytest.mark.asyncio
async def test_phase2_retrieve_allows_premium_agent_global_on_dev_override():
    from app.routers import retrieve_memory, MemoryRetrieveRequest
    from app.services.pgvector_search import VectorSearchResult

    user_id = str(uuid4())
    org_id = str(uuid4())
    agent_hash = "agent_abc"

    payload = MemoryRetrieveRequest(
        user_id=user_id,
        org_id=org_id,
        agent_hash=agent_hash,
        query="hello",
        limit=10,
        use_vector_search=True,
    )

    request = _make_request({"x-is-dev-override": "true"})
    session = AsyncMock()

    agent_premium = VectorSearchResult(
        memory_id="m_agent_premium",
        content="agent_premium",
        similarity=0.6,
        metadata={"tier": "premium", "record_user_id": None, "record_agent_hash": agent_hash},
    )

    async def _search_similar_side_effect(*, require_user_id_null=False, **kwargs):
        if require_user_id_null:
            return [agent_premium]
        return []

    with (
        patch("app.routers.semantic_cache.get", return_value=None),
        patch("app.routers.semantic_cache.set", return_value=None),
        patch("app.routers.embedding_cache.get", return_value=None),
        patch("app.routers.embedding_cache.set", return_value=None),
        patch("app.routers.embeddings_generator.generate", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]])),
        patch("app.routers.pgvector_search.check_pgvector_available", new=AsyncMock(return_value=True)),
        patch("app.routers.pgvector_search.search_similar", new=AsyncMock(side_effect=_search_similar_side_effect)),
    ):
        results = await retrieve_memory(payload=payload, request=request, session=session)

    assert any(r.content == "agent_premium" for r in results)
    premium = next(r for r in results if r.content == "agent_premium")
    assert premium.scope == "agent_global"
    assert premium.tier == "premium"


@pytest.mark.asyncio
async def test_phase2_retrieve_hybrid_mode_reranks_using_hash_sphere_signals():
    from app.routers import retrieve_memory, MemoryRetrieveRequest
    from app.services.pgvector_search import VectorSearchResult

    user_id = str(uuid4())
    org_id = str(uuid4())
    agent_hash = "agent_abc"

    payload = MemoryRetrieveRequest(
        user_id=user_id,
        org_id=org_id,
        agent_hash=agent_hash,
        query="hello",
        limit=10,
        use_vector_search=True,
        retrieval_mode="hybrid",
    )

    request = _make_request({"x-is-dev-override": "true"})
    session = AsyncMock()

    # A wins on embedding similarity, B should win after hybrid scoring.
    a = VectorSearchResult(
        memory_id="m_a",
        content="A",
        similarity=0.95,
        hash="hash_a",
        xyz=(0.1, 0.1, 0.1),
        resonance_score=None,
        metadata={"record_user_id": user_id, "record_agent_hash": None},
    )
    b = VectorSearchResult(
        memory_id="m_b",
        content="B",
        similarity=0.10,
        hash="hash_b",
        xyz=(0.9, 0.9, 0.9),
        resonance_score=None,
        metadata={"record_user_id": user_id, "record_agent_hash": None},
    )

    async def _search_similar_side_effect(*, user_id=None, require_agent_hash_null=False, **kwargs):
        if user_id is not None and require_agent_hash_null:
            return [a, b]
        return []

    class _QueryCoords:
        x = 0.0
        y = 0.0
        z = 0.0
        resonance_score = 0.0

    def _resonance_side_effect(qh, mh):
        if mh == "hash_b":
            return 1.0
        return 0.0

    def _proximity_side_effect(q_xyz, m_xyz):
        if m_xyz == (0.9, 0.9, 0.9):
            return 1.0
        return 0.0

    def _res_fn_side_effect(xyz):
        # Make A look far in resonance function space, B close.
        if xyz == (0.1, 0.1, 0.1):
            return 6.0
        return 0.0

    def _anchor_energy_side_effect(q_xyz, m_xyz):
        if tuple(m_xyz.tolist()) == (0.9, 0.9, 0.9):
            return 1.0
        return 0.0

    with (
        patch("app.routers.semantic_cache.get", return_value=None),
        patch("app.routers.semantic_cache.set", return_value=None),
        patch("app.routers.embedding_cache.get", return_value=None),
        patch("app.routers.embedding_cache.set", return_value=None),
        patch("app.routers.embeddings_generator.generate", new=AsyncMock(return_value=[[0.1, 0.2, 0.3]])),
        patch("app.routers.pgvector_search.check_pgvector_available", new=AsyncMock(return_value=True)),
        patch("app.routers.pgvector_search.search_similar", new=AsyncMock(side_effect=_search_similar_side_effect)),
        patch("app.routers.ResonanceHasher.compute_full_coordinates", return_value=_QueryCoords()),
        patch("app.routers.ResonanceHasher.hash_text", return_value="query_hash"),
        patch("app.routers.resonance_hasher.calculate_resonance", side_effect=_resonance_side_effect),
        patch("app.routers.ResonanceHasher.calculate_proximity_score", side_effect=_proximity_side_effect),
        patch("app.routers.ResonanceHasher.calculate_resonance_function", side_effect=_res_fn_side_effect),
        patch("app.routers.ResonanceHasher.calculate_anchor_energy", side_effect=_anchor_energy_side_effect),
    ):
        results = await retrieve_memory(payload=payload, request=request, session=session)

    assert len(results) >= 2
    assert results[0].content == "B"
