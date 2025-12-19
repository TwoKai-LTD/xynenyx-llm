"""Anthropic provider implementation (placeholder for future)."""
# TODO: Implement Anthropic provider when API key is available
# This will follow the same pattern as OpenAI provider
# - Use langchain-anthropic for ChatAnthropic
# - Implement complete(), stream(), embed() methods
# - Add to provider router when ready

from app.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic provider (placeholder)."""

    def __init__(self):
        super().__init__(name="anthropic", timeout=60)
        # TODO: Initialize Anthropic client when API key is available

    async def complete(self, messages, model=None, temperature=0.7, max_tokens=None):
        """TODO: Implement Anthropic completion."""
        raise NotImplementedError("Anthropic provider not yet implemented")

    async def stream(self, messages, model=None, temperature=0.7, max_tokens=None):
        """TODO: Implement Anthropic streaming."""
        raise NotImplementedError("Anthropic provider not yet implemented")

    async def embed(self, text, model=None):
        """TODO: Implement Anthropic embeddings."""
        raise NotImplementedError("Anthropic provider not yet implemented")

    async def health_check(self) -> bool:
        """TODO: Implement Anthropic health check."""
        return False

