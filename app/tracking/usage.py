"""Usage tracking for LLM requests."""
from typing import Dict, Any, Optional
from supabase import create_client, Client
from app.config import settings
from app.providers.base import CompletionUsage


class UsageTracker:
    """Tracks LLM usage and costs to Supabase."""

    def __init__(self):
        """Initialize usage tracker with Supabase client."""
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        self.cost_rates = settings.cost_rates

    def _calculate_cost(
        self,
        provider: str,
        model: str,
        usage: CompletionUsage,
    ) -> float:
        """
        Calculate cost based on usage and model rates.

        Args:
            provider: Provider name (e.g., "openai")
            model: Model name (e.g., "gpt-4o-mini")
            usage: Usage metadata

        Returns:
            Cost in USD
        """
        rates = self.cost_rates.get(model, {"input": 0.0, "output": 0.0})

        input_cost = (usage.prompt_tokens / 1000) * rates.get("input", 0.0)
        output_cost = (usage.completion_tokens / 1000) * rates.get("output", 0.0)

        return input_cost + output_cost

    async def track(
        self,
        user_id: str,
        conversation_id: Optional[str],
        provider: str,
        model: str,
        usage: CompletionUsage,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track LLM usage to Supabase.

        Args:
            user_id: User ID (from request header)
            conversation_id: Optional conversation ID
            provider: Provider name
            model: Model name
            usage: Usage metadata
            metadata: Additional metadata to store
        """
        cost = self._calculate_cost(provider, model, usage)

        try:
            # Note: total_tokens is a generated column, so we don't insert it
            self.client.table("llm_usage").insert(
                {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "provider": provider,
                    "model": model,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cost_usd": cost,
                    "metadata": metadata or {},
                }
            ).execute()
        except Exception as e:
            # Log error but don't fail the request
            # In production, use proper logging
            print(f"Error tracking usage: {e}")

