"""Integration tests for API endpoints."""
import pytest
import json
from unittest.mock import patch, AsyncMock, Mock

from app.providers.base import CompletionResponse, CompletionUsage, EmbeddingResponse


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_ready_endpoint(client):
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_list_providers(client):
    """Test listing providers endpoint."""
    with patch("app.routers.providers.provider_router") as mock_router:
        mock_router.list_providers.return_value = {
            "openai": {
                "id": "openai",
                "name": "openai",
                "enabled": True,
                "models": ["gpt-4o", "gpt-4o-mini"],
            }
        }
        mock_router.check_all_health = AsyncMock(return_value={"openai": True})

        response = client.get("/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) > 0


@pytest.mark.asyncio
async def test_get_provider(client):
    """Test getting specific provider endpoint."""
    with patch("app.routers.providers.provider_router") as mock_router:
        mock_provider = Mock()
        mock_provider.name = "openai"
        mock_provider.get_supported_models.return_value = ["gpt-4o", "gpt-4o-mini"]
        mock_router.get_provider.return_value = mock_provider
        mock_router.check_provider_health = AsyncMock(return_value=True)

        response = client.get("/providers/openai")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "openai"


@pytest.mark.asyncio
async def test_complete_endpoint_missing_user_id(client):
    """Test completion endpoint requires user ID."""
    response = client.post(
        "/complete",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 400
    assert "X-User-ID" in response.json()["detail"]


@pytest.mark.asyncio
async def test_complete_endpoint(client, test_user_id):
    """Test completion endpoint."""
    with patch("app.routers.completions.provider_router") as mock_router:
        mock_provider = Mock()
        mock_provider.name = "openai"
        mock_provider.complete = AsyncMock(
            return_value=CompletionResponse(
                content="Test response",
                usage=CompletionUsage(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                ),
                model="gpt-4o-mini",
                metadata={},
            )
        )
        mock_router.get_provider.return_value = mock_provider

        response = client.post(
            "/complete",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"X-User-ID": test_user_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "provider" in data
        assert data["provider"] == "openai"


@pytest.mark.asyncio
async def test_embedding_endpoint_missing_user_id(client):
    """Test embedding endpoint requires user ID."""
    response = client.post(
        "/embeddings",
        json={"text": "test text"},
    )
    assert response.status_code == 400
    assert "X-User-ID" in response.json()["detail"]


@pytest.mark.asyncio
async def test_embedding_endpoint(client, test_user_id):
    """Test embedding endpoint."""
    with patch("app.routers.embeddings.provider_router") as mock_router:
        mock_provider = Mock()
        mock_provider.name = "openai"
        mock_provider.embed = AsyncMock(
            return_value=EmbeddingResponse(
                embedding=[0.1] * 1536,
                model="text-embedding-ada-002",
                usage=CompletionUsage(
                    prompt_tokens=10,
                    completion_tokens=0,
                    total_tokens=10,
                ),
                metadata={},
            )
        )
        mock_router.get_provider.return_value = mock_provider

        response = client.post(
            "/embeddings",
            json={"text": "test text"},
            headers={"X-User-ID": test_user_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "provider" in data
        assert data["provider"] == "openai"
        assert len(data["embedding"]) == 1536

