"""Google Gemini provider implementation (placeholder for future)."""
# TODO: Implement Gemini provider when API key is available
# This will follow the same pattern as OpenAI provider
# - Use langchain-google-genai for ChatGoogleGenerativeAI
# - Implement complete(), stream(), embed() methods
# - Add to provider router when ready

from app.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini provider (placeholder)."""

    def __init__(self):
        super().__init__(name="gemini", timeout=60)
        # TODO: Initialize Gemini client when API key is available

    async def complete(self, messages, model=None, temperature=0.7, max_tokens=None):
        """TODO: Implement Gemini completion."""
        raise NotImplementedError("Gemini provider not yet implemented")

    async def stream(self, messages, model=None, temperature=0.7, max_tokens=None):
        """TODO: Implement Gemini streaming."""
        raise NotImplementedError("Gemini provider not yet implemented")

    async def embed(self, text, model=None):
        """TODO: Implement Gemini embeddings."""
        raise NotImplementedError("Gemini provider not yet implemented")

    async def health_check(self) -> bool:
        """TODO: Implement Gemini health check."""
        return False

