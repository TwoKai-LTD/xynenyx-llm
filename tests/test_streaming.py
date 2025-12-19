"""Tests for streaming functionality."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from app.providers.base import StreamChunk, CompletionUsage
from app.tracking.callbacks import StreamingCallback


@pytest.mark.asyncio
async def test_streaming_callback():
    """Test streaming callback handler."""
    queue = asyncio.Queue()
    callback = StreamingCallback(queue)

    # Test new token
    await callback.on_llm_new_token("Hello")
    chunk = await queue.get()
    assert chunk.type == "token"
    assert chunk.content == "Hello"

    # Test end with usage
    usage_metadata = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    llm_result = Mock()
    llm_result.llm_output = {"token_usage": usage_metadata}

    await callback.on_llm_end(llm_result)
    chunk = await queue.get()
    assert chunk.type == "end"
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 20
    assert chunk.usage.total_tokens == 30

    # Test error
    error = ValueError("Test error")
    await callback.on_llm_error(error)
    chunk = await queue.get()
    assert chunk.type == "error"
    assert "Test error" in chunk.content


@pytest.mark.asyncio
async def test_streaming_callback_no_usage():
    """Test streaming callback when no usage metadata."""
    queue = asyncio.Queue()
    callback = StreamingCallback(queue)

    llm_result = Mock()
    llm_result.llm_output = {}

    await callback.on_llm_end(llm_result)
    chunk = await queue.get()
    assert chunk.type == "end"
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 0
    assert chunk.usage.completion_tokens == 0

