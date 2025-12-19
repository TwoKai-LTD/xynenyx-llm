"""Embedding endpoints for LLM service."""
from fastapi import APIRouter, Header, HTTPException
from typing import Optional

from app.schemas.embeddings import EmbeddingRequest, EmbeddingResponse
from app.providers.router import ProviderRouter
from app.tracking.usage import UsageTracker

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# Initialize dependencies
provider_router = ProviderRouter()
usage_tracker = UsageTracker()


@router.post("", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
):
    """
    Generate embeddings for text.

    Requires X-User-ID header for usage tracking.
    """
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")

    try:
        # Get provider
        provider = provider_router.get_provider(request.provider)

        # Generate embedding
        response = await provider.embed(
            text=request.text,
            model=request.model,
        )

        # Track usage
        await usage_tracker.track(
            user_id=x_user_id,
            conversation_id=x_conversation_id,
            provider=provider.name,
            model=response.model,
            usage=response.usage,
            metadata=response.metadata,
        )

        return EmbeddingResponse(
            provider=provider.name,
            embedding=response.embedding,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            metadata=response.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

