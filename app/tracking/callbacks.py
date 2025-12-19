"""LangChain callbacks for streaming."""
import asyncio
from typing import Any
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from app.providers.base import StreamChunk, CompletionUsage


class StreamingCallback(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, queue: asyncio.Queue):
        """
        Initialize streaming callback.

        Args:
            queue: Async queue to put chunks into
        """
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

