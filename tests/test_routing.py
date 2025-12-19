"""Tests for provider routing logic."""
import pytest
from unittest.mock import patch

from app.providers.router import ProviderRouter


def test_provider_router_list_providers():
    """Test listing all providers."""
    with patch("app.providers.router.settings") as mock_settings:
        mock_settings.openai_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_enabled = False
        mock_settings.google_enabled = False

        router = ProviderRouter()
        providers = router.list_providers()
        assert len(providers) > 0
        assert "openai" in providers


@pytest.mark.asyncio
async def test_provider_router_health_check():
    """Test provider health checking."""
    with patch("app.providers.router.settings") as mock_settings:
        mock_settings.openai_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_enabled = False
        mock_settings.google_enabled = False

        router = ProviderRouter()
        health = await router.check_provider_health("openai")
        assert isinstance(health, bool)


@pytest.mark.asyncio
async def test_provider_router_check_all_health():
    """Test checking health of all providers."""
    with patch("app.providers.router.settings") as mock_settings:
        mock_settings.openai_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_enabled = False
        mock_settings.google_enabled = False

        router = ProviderRouter()
        health_status = await router.check_all_health()
        assert "openai" in health_status
        assert isinstance(health_status["openai"], bool)


def test_provider_router_invalid_provider():
    """Test getting invalid provider raises error."""
    with patch("app.providers.router.settings") as mock_settings:
        mock_settings.openai_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_enabled = False
        mock_settings.google_enabled = False

        router = ProviderRouter()
        with pytest.raises(ValueError):
            router.get_provider("invalid-provider")

