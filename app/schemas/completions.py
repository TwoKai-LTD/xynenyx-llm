"""Pydantic schemas for completion requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Message(BaseModel):
    """A chat message."""

    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class CompletionRequest(BaseModel):
    """Request for a completion."""

    messages: List[Message] = Field(..., description="List of chat messages")
    provider: Optional[str] = Field(None, description="Provider ID (e.g., 'openai'). Defaults to first available.")
    model: Optional[str] = Field(None, description="Model name. Defaults to provider default.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")


class CompletionResponse(BaseModel):
    """Response from a completion request."""

    provider: str = Field(..., description="Provider used")
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    usage: Dict[str, int] = Field(..., description="Token usage (prompt_tokens, completion_tokens, total_tokens)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StreamChunk(BaseModel):
    """A chunk from a streaming completion."""

    type: str = Field(..., description="Chunk type: 'token', 'end', or 'error'")
    content: str = Field(default="", description="Chunk content (for token type)")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage (for end type)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

