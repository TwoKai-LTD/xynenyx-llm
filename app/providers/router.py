"""Provider router for selecting and managing LLM providers."""
from typing import Dict, Optional
from app.providers.base import BaseProvider
from app.providers.openai import OpenAIProvider
from app.providers.anthropic import AnthropicProvider
from app.providers.gemini import GeminiProvider
from app.config import settings


class ProviderRouter:
    """Router for managing and selecting LLM providers."""

    def __init__(self):
        """Initialize provider router with available providers."""
        self.providers: Dict[str, BaseProvider] = {}

        # Initialize OpenAI (primary, always enabled if configured)
        if settings.openai_enabled and settings.openai_api_key:
            self.providers["openai"] = OpenAIProvider()

        # Initialize Anthropic (optional, future)
        if settings.anthropic_enabled and settings.anthropic_api_key:
            # TODO: Uncomment when Anthropic provider is implemented
            # self.providers["anthropic"] = AnthropicProvider()
            pass

        # Initialize Gemini (optional, future)
        if settings.google_enabled and settings.google_api_key:
            # TODO: Uncomment when Gemini provider is implemented
            # self.providers["gemini"] = GeminiProvider()
            pass

    def get_provider(self, provider_id: str | None = None) -> BaseProvider:
        """
        Get a provider by ID, or return the default provider.

        Args:
            provider_id: Provider ID (e.g., "openai"). If None, returns default.

        Returns:
            BaseProvider instance

        Raises:
            ValueError: If provider not found or not enabled
        """
        if provider_id is None:
            # Return first available provider (OpenAI for now)
            if "openai" in self.providers:
                return self.providers["openai"]
            raise ValueError("No providers available")

        if provider_id not in self.providers:
            raise ValueError(f"Provider '{provider_id}' not found or not enabled")

        return self.providers[provider_id]

    def list_providers(self) -> Dict[str, Dict]:
        """
        List all available providers with their status.

        Returns:
            Dict mapping provider IDs to their info
        """
        result = {}
        for provider_id, provider in self.providers.items():
            result[provider_id] = {
                "id": provider_id,
                "name": provider.name,
                "enabled": True,
                "models": provider.get_supported_models(),
                "healthy": False,  # Will be checked separately
            }
        return result

    async def check_provider_health(self, provider_id: str) -> bool:
        """
        Check health of a specific provider.

        Args:
            provider_id: Provider ID to check

        Returns:
            True if healthy, False otherwise
        """
        if provider_id not in self.providers:
            return False

        try:
            return await self.providers[provider_id].health_check()
        except Exception:
            return False

    async def check_all_health(self) -> Dict[str, bool]:
        """
        Check health of all providers.

        Returns:
            Dict mapping provider IDs to health status
        """
        health_status = {}
        for provider_id in self.providers:
            health_status[provider_id] = await self.check_provider_health(provider_id)
        return health_status

