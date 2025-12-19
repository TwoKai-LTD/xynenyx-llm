"""OpenAI provider implementation using LangChain."""
import asyncio
from typing import List, Dict, Any, AsyncIterator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from app.providers.base import (
    BaseProvider,
    CompletionResponse,
    CompletionUsage,
    EmbeddingResponse,
    StreamChunk,
)
from app.config import settings


class OpenAIStreamingHandler(AsyncCallbackHandler):
    """Callback handler for OpenAI streaming."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.usage: CompletionUsage | None = None

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        await self.queue.put(
            StreamChunk(
                type="token",
                content=token,
            )
        )

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes generating."""
        usage_metadata = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.usage = CompletionUsage(
            prompt_tokens=usage_metadata.get("prompt_tokens", 0),
            completion_tokens=usage_metadata.get("completion_tokens", 0),
            total_tokens=usage_metadata.get("total_tokens", 0),
        )
        await self.queue.put(
            StreamChunk(
                type="end",
                usage=self.usage,
            )
        )

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error."""
        await self.queue.put(
            StreamChunk(
                type="error",
                content=str(error),
            )
        )


class OpenAIProvider(BaseProvider):
    """OpenAI provider using LangChain."""

    def __init__(self):
        super().__init__(name="openai", timeout=settings.request_timeout)
        self.api_key = settings.openai_api_key
        self.default_model = settings.default_model
        self.default_embedding_model = settings.default_embedding_model

        # Initialize clients
        self.chat_client = ChatOpenAI(
            api_key=self.api_key,
            model=self.default_model,
            temperature=0.7,
            timeout=self.timeout,
        )
        self.embedding_client = OpenAIEmbeddings(
            api_key=self.api_key,
            model=self.default_embedding_model,
        )

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List:
        """Convert message dicts to LangChain message objects."""
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:  # user or default
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResponse:
        """Generate a synchronous completion."""
        try:
            # Create client with specific model if provided
            client = ChatOpenAI(
                api_key=self.api_key,
                model=model or self.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )

            langchain_messages = self._convert_messages(messages)
            response = await client.ainvoke(langchain_messages)

            # Extract usage metadata
            response_metadata = getattr(response, "response_metadata", {})
            usage_metadata = response_metadata.get("token_usage", {})
            usage = CompletionUsage(
                prompt_tokens=usage_metadata.get("prompt_tokens", 0),
                completion_tokens=usage_metadata.get("completion_tokens", 0),
                total_tokens=usage_metadata.get("total_tokens", 0),
            )

            # Get content from response
            content = response.content if hasattr(response, "content") else str(response)

            return CompletionResponse(
                content=content,
                usage=usage,
                model=model or self.default_model,
                metadata={
                    "finish_reason": response_metadata.get("finish_reason"),
                },
            )
        except Exception as e:
            raise ValueError(f"OpenAI completion error: {str(e)}") from e

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion."""
        try:
            # Create client with streaming enabled
            client = ChatOpenAI(
                api_key=self.api_key,
                model=model or self.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                streaming=True,
            )

            langchain_messages = self._convert_messages(messages)
            usage_data: CompletionUsage | None = None
            last_chunk = None

            # Stream directly from LangChain (no callbacks for async streaming)
            try:
                async for chunk in client.astream(langchain_messages):
                    # Extract token from chunk
                    if hasattr(chunk, "content") and chunk.content:
                        yield StreamChunk(
                            type="token",
                            content=chunk.content,
                        )
                    last_chunk = chunk
                    
                    # Try to extract usage metadata from chunk
                    if hasattr(chunk, "response_metadata"):
                        token_usage = chunk.response_metadata.get("token_usage", {})
                        if token_usage:
                            usage_data = CompletionUsage(
                                prompt_tokens=token_usage.get("prompt_tokens", 0),
                                completion_tokens=token_usage.get("completion_tokens", 0),
                                total_tokens=token_usage.get("total_tokens", 0),
                            )

                # Try to get usage from last chunk if not already captured
                if not usage_data and last_chunk:
                    if hasattr(last_chunk, "response_metadata"):
                        token_usage = last_chunk.response_metadata.get("token_usage", {})
                        if token_usage:
                            usage_data = CompletionUsage(
                                prompt_tokens=token_usage.get("prompt_tokens", 0),
                                completion_tokens=token_usage.get("completion_tokens", 0),
                                total_tokens=token_usage.get("total_tokens", 0),
                            )

                # Yield final chunk with usage (or empty if not available)
                yield StreamChunk(
                    type="end",
                    usage=usage_data or CompletionUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                )

            except Exception as e:
                yield StreamChunk(
                    type="error",
                    content=str(e),
                )

        except Exception as e:
            yield StreamChunk(
                type="error",
                content=str(e),
            )

    async def embed(
        self,
        text: str,
        model: str | None = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for text."""
        try:
            # Use default embedding model if not specified
            embedding_model = model or self.default_embedding_model

            # Create embedding client with specific model
            client = OpenAIEmbeddings(
                api_key=self.api_key,
                model=embedding_model,
            )

            # Generate embedding
            embedding = await client.aembed_query(text)

            # Estimate token usage (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(text) // 4

            usage = CompletionUsage(
                prompt_tokens=estimated_tokens,
                completion_tokens=0,
                total_tokens=estimated_tokens,
            )

            return EmbeddingResponse(
                embedding=embedding,
                model=embedding_model,
                usage=usage,
                metadata={},
            )
        except Exception as e:
            raise ValueError(f"OpenAI embedding error: {str(e)}") from e

    async def health_check(self) -> bool:
        """Check if OpenAI provider is healthy."""
        try:
            # Simple health check: try to list models
            # For now, we'll just check if API key is set
            if not self.api_key:
                return False

            # Could make a lightweight API call here, but for now just check key
            return True
        except Exception:
            return False

    def get_supported_models(self) -> List[str]:
        """Get list of supported OpenAI models."""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "text-embedding-ada-002",
        ]

