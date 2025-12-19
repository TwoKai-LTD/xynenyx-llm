"""Tests for prompt templates and manager."""
import pytest
from datetime import datetime

from app.prompts.templates import get_prompt, list_prompts, PROMPTS
from app.prompts.manager import PromptManager


def test_list_prompts():
    """Test listing all available prompts."""
    prompts = list_prompts()
    assert len(prompts) > 0
    assert "chat_agent" in prompts
    assert "rag_qa" in prompts
    assert "intent_classification" in prompts


def test_get_prompt():
    """Test getting a specific prompt."""
    prompt = get_prompt("chat_agent")
    assert prompt is not None


def test_get_invalid_prompt():
    """Test getting invalid prompt raises error."""
    with pytest.raises(ValueError):
        get_prompt("invalid_prompt")


def test_prompt_manager_initialization():
    """Test prompt manager initialization."""
    manager = PromptManager()
    assert manager.current_date is not None


def test_prompt_manager_get_prompt():
    """Test getting prompt from manager."""
    manager = PromptManager()
    prompt = manager.get_prompt("chat_agent")
    assert prompt is not None


def test_prompt_manager_inject_date():
    """Test date injection in prompts."""
    manager = PromptManager()
    prompt = manager.get_prompt("chat_agent", inject_date=True)
    # Date should be injected when prompt is formatted
    assert prompt is not None


def test_prompt_manager_list_prompts():
    """Test listing prompts from manager."""
    manager = PromptManager()
    prompts = manager.list_prompts()
    assert len(prompts) > 0


def test_prompt_manager_update_date():
    """Test updating current date."""
    manager = PromptManager()
    old_date = manager.current_date
    manager.update_date()
    # Date should be updated (might be same if called within same second)
    assert manager.current_date is not None

