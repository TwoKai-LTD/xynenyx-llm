"""Pydantic schemas for embedding requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class EmbeddingRequest(BaseModel):
    """Request for embeddings."""

    text: str = Field(..., description="Text to generate embeddings for")
    provider: Optional[str] = Field(None, description="Provider ID (e.g., 'openai'). Defaults to first available.")
    model: Optional[str] = Field(None, description="Embedding model name. Defaults to provider default.")


class EmbeddingResponse(BaseModel):
    """Response from an embedding request."""

    provider: str = Field(..., description="Provider used")
    embedding: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used")
    usage: Dict[str, int] = Field(..., description="Token usage (prompt_tokens, completion_tokens, total_tokens)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

