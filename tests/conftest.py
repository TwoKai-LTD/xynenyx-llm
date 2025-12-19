"""Pytest configuration and fixtures."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from typing import AsyncIterator

from app.main import app
from app.providers.base import BaseProvider, CompletionResponse, CompletionUsage, EmbeddingResponse, StreamChunk
from app.config import settings


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider for testing."""
    provider = Mock(spec=BaseProvider)
    provider.name = "openai"
    provider.default_model = "gpt-4o-mini"

    # Mock complete method
    async def mock_complete(*args, **kwargs):
        return CompletionResponse(
            content="Test response",
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
            model="gpt-4o-mini",
            metadata={},
        )

    provider.complete = AsyncMock(side_effect=mock_complete)

    # Mock stream method
    async def mock_stream(*args, **kwargs) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(type="token", content="Test")
        yield StreamChunk(type="token", content=" response")
        yield StreamChunk(
            type="end",
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )

    provider.stream = AsyncMock(return_value=mock_stream())

    # Mock embed method
    async def mock_embed(*args, **kwargs):
        return EmbeddingResponse(
            embedding=[0.1] * 1536,
            model="text-embedding-ada-002",
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=0,
                total_tokens=10,
            ),
            metadata={},
        )

    provider.embed = AsyncMock(side_effect=mock_embed)
    provider.health_check = AsyncMock(return_value=True)
    provider.get_supported_models = Mock(return_value=["gpt-4o", "gpt-4o-mini"])

    return provider


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    mock_client = Mock()
    mock_table = Mock()
    mock_table.insert = Mock(return_value=mock_table)
    mock_table.execute = Mock(return_value=Mock())
    mock_client.table = Mock(return_value=mock_table)
    return mock_client


@pytest.fixture
def test_user_id():
    """Test user ID for headers."""
    return "test-user-123"


@pytest.fixture
def test_conversation_id():
    """Test conversation ID for headers."""
    return "test-conversation-456"


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
    ]
