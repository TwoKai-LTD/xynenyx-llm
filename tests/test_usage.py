"""Tests for usage tracking."""
import pytest
from unittest.mock import Mock, patch

from app.tracking.usage import UsageTracker
from app.providers.base import CompletionUsage


def test_usage_tracker_initialization():
    """Test usage tracker initialization."""
    with patch("app.tracking.usage.settings") as mock_settings:
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_role_key = "test-key"
        mock_settings.cost_rates = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        tracker = UsageTracker()
        assert tracker.client is not None


def test_usage_tracker_calculate_cost():
    """Test cost calculation."""
    with patch("app.tracking.usage.settings") as mock_settings:
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_role_key = "test-key"
        mock_settings.cost_rates = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        tracker = UsageTracker()
        usage = CompletionUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        cost = tracker._calculate_cost("openai", "gpt-4o-mini", usage)
        # 1000 * 0.00015 / 1000 + 500 * 0.0006 / 1000
        # = 0.15 + 0.3 = 0.45
        expected_cost = (1000 / 1000 * 0.00015) + (500 / 1000 * 0.0006)
        assert abs(cost - expected_cost) < 0.0001


@pytest.mark.asyncio
async def test_usage_tracker_track():
    """Test tracking usage to Supabase."""
    with patch("app.tracking.usage.settings") as mock_settings:
        mock_settings.supabase_url = "https://test.supabase.co"
        mock_settings.supabase_service_role_key = "test-key"
        mock_settings.cost_rates = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        tracker = UsageTracker()
        # Mock the Supabase client
        mock_table = Mock()
        mock_table.insert = Mock(return_value=mock_table)
        mock_table.execute = Mock(return_value=Mock())
        tracker.client = Mock()
        tracker.client.table = Mock(return_value=mock_table)

        usage = CompletionUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        await tracker.track(
            user_id="test-user",
            conversation_id="test-conv",
            provider="openai",
            model="gpt-4o-mini",
            usage=usage,
        )

        # Verify insert was called
        mock_table.insert.assert_called_once()
        mock_table.execute.assert_called_once()

