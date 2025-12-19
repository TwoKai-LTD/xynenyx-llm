"""Completion endpoints for LLM service."""
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import json

from app.schemas.completions import CompletionRequest, CompletionResponse, StreamChunk
from app.providers.router import ProviderRouter
from app.tracking.usage import UsageTracker
from app.providers.base import CompletionUsage

router = APIRouter(prefix="/complete", tags=["completions"])

# Initialize dependencies
provider_router = ProviderRouter()
usage_tracker = UsageTracker()


@router.post("", response_model=CompletionResponse)
async def complete(
    request: CompletionRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
):
    """
    Generate a synchronous completion.

    Requires X-User-ID header for usage tracking.
    """
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")

    try:
        # Get provider
        provider = provider_router.get_provider(request.provider)

        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Generate completion
        response = await provider.complete(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
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

        return CompletionResponse(
            provider=provider.name,
            content=response.content,
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


@router.post("/stream")
async def complete_stream(
    request: CompletionRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
):
    """
    Generate a streaming completion (Server-Sent Events).

    Requires X-User-ID header for usage tracking.
    """
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")

    async def generate_stream():
        """Generate SSE stream."""
        usage_data: CompletionUsage | None = None
        provider_name: str | None = None
        model_name: str | None = None

        try:
            # Get provider
            provider = provider_router.get_provider(request.provider)
            provider_name = provider.name

            # Convert messages to dict format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            # Stream completion
            async for chunk in provider.stream(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                # Track model name from first chunk
                if model_name is None and chunk.type == "token":
                    model_name = request.model or provider.default_model

                # Track usage from end chunk
                if chunk.type == "end" and chunk.usage:
                    usage_data = chunk.usage
                    model_name = request.model or provider.default_model

                # Format as SSE
                chunk_dict = {
                    "type": chunk.type,
                    "content": chunk.content,
                }
                if chunk.usage:
                    chunk_dict["usage"] = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
                if chunk.metadata:
                    chunk_dict["metadata"] = chunk.metadata

                yield f"data: {json.dumps(chunk_dict)}\n\n"

                # Stop on error or end
                if chunk.type in ("error", "end"):
                    break

            # Track usage after streaming completes
            if usage_data and provider_name and model_name:
                await usage_tracker.track(
                    user_id=x_user_id,
                    conversation_id=x_conversation_id,
                    provider=provider_name,
                    model=model_name,
                    usage=usage_data,
                )

        except ValueError as e:
            error_chunk = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_chunk)}\n\n"
        except Exception as e:
            error_chunk = {"type": "error", "content": f"Internal server error: {str(e)}"}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

