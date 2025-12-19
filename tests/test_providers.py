"""Tests for provider implementations."""
import pytest
from unittest.mock import patch, AsyncMock

from app.providers.openai import OpenAIProvider
from app.providers.router import ProviderRouter
from app.providers.base import CompletionUsage


@pytest.mark.asyncio
async def test_openai_provider_initialization():
    """Test OpenAI provider initialization."""
    with patch("app.providers.openai.settings") as mock_settings:
        mock_settings.openai_api_key = "test-key"
        mock_settings.default_model = "gpt-4o-mini"
        mock_settings.default_embedding_model = "text-embedding-ada-002"
        mock_settings.request_timeout = 60

        provider = OpenAIProvider()
        assert provider.name == "openai"
        assert provider.default_model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openai_provider_health_check():
    """Test OpenAI provider health check."""
    with patch("app.providers.openai.settings") as mock_settings:
        mock_settings.openai_api_key = "test-key"
        mock_settings.default_model = "gpt-4o-mini"
        mock_settings.default_embedding_model = "text-embedding-ada-002"
        mock_settings.request_timeout = 60

        provider = OpenAIProvider()
        # Health check should return True if API key is set
        result = await provider.health_check()
        assert result is True


def test_openai_provider_supported_models():
    """Test OpenAI provider supported models."""
    with patch("app.providers.openai.settings") as mock_settings:
        mock_settings.openai_api_key = "test-key"
        mock_settings.default_model = "gpt-4o-mini"
        mock_settings.default_embedding_model = "text-embedding-ada-002"
        mock_settings.request_timeout = 60

        provider = OpenAIProvider()
        models = provider.get_supported_models()
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "text-embedding-ada-002" in models


@pytest.mark.asyncio
async def test_provider_router_initialization():
    """Test provider router initialization."""
    with patch("app.providers.router.settings") as mock_settings:
        mock_settings.openai_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_enabled = False
        mock_settings.google_enabled = False

        router = ProviderRouter()
        assert "openai" in router.providers


@pytest.mark.asyncio
async def test_provider_router_get_provider():
    """Test getting a provider from router."""
    with patch("app.providers.router.settings") as mock_settings:
        mock_settings.openai_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_enabled = False
        mock_settings.google_enabled = False

        router = ProviderRouter()
        provider = router.get_provider("openai")
        assert provider.name == "openai"


@pytest.mark.asyncio
async def test_provider_router_get_default_provider():
    """Test getting default provider."""
    with patch("app.providers.router.settings") as mock_settings:
        mock_settings.openai_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_enabled = False
        mock_settings.google_enabled = False

        router = ProviderRouter()
        provider = router.get_provider(None)
        assert provider.name == "openai"


def test_provider_router_list_providers():
    """Test listing providers."""
    with patch("app.providers.router.settings") as mock_settings:
        mock_settings.openai_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_enabled = False
        mock_settings.google_enabled = False

        router = ProviderRouter()
        providers = router.list_providers()
        assert "openai" in providers
        assert providers["openai"]["name"] == "openai"

