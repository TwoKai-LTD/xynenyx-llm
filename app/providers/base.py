"""Abstract base class for LLM providers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator
from pydantic import BaseModel


class CompletionUsage(BaseModel):
    """Usage metadata from a completion request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    """Response from a completion request."""

    content: str
    usage: CompletionUsage
    model: str
    metadata: Dict[str, Any] = {}


class EmbeddingResponse(BaseModel):
    """Response from an embedding request."""

    embedding: List[float]
    model: str
    usage: CompletionUsage
    metadata: Dict[str, Any] = {}


class StreamChunk(BaseModel):
    """A chunk from a streaming completion."""

    type: str  # "token", "end", "error"
    content: str = ""
    usage: CompletionUsage | None = None
    metadata: Dict[str, Any] = {}


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, timeout: int = 60):
        """
        Initialize the provider.

        Args:
            name: Provider name (e.g., "openai")
            timeout: Request timeout in seconds
        """
        self.name = name
        self.timeout = timeout

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResponse:
        """
        Generate a synchronous completion.

        Args:
            messages: List of message dicts with "role" and "content"
            model: Model name (optional, uses default if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            CompletionResponse with content, usage, and metadata
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate a streaming completion.

        Args:
            messages: List of message dicts with "role" and "content"
            model: Model name (optional, uses default if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            StreamChunk objects with tokens or metadata
        """
        pass

    @abstractmethod
    async def embed(
        self,
        text: str,
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed
            model: Embedding model name (optional)

        Returns:
            EmbeddingResponse with embedding vector and usage
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and available.

        Returns:
            True if provider is healthy, False otherwise
        """
        pass

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported models for this provider.

        Returns:
            List of model names
        """
        return []

