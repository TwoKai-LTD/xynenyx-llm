"""Prompt template manager with versioning and variable substitution."""
from datetime import datetime
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from app.prompts.templates import get_prompt, PROMPTS


class PromptManager:
    """Manages prompt templates with versioning and variable substitution."""

    def __init__(self):
        """Initialize prompt manager."""
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def get_prompt(
        self,
        name: str,
        variables: Dict[str, Any] | None = None,
        inject_date: bool = True,
    ) -> ChatPromptTemplate:
        """
        Get a prompt template with variable substitution.

        Args:
            name: Prompt template name
            variables: Variables to substitute
            inject_date: Whether to inject current date

        Returns:
            ChatPromptTemplate with variables substituted
        """
        prompt = get_prompt(name)

        # Prepare variables
        vars_dict = variables or {}
        if inject_date and "current_date" not in vars_dict:
            vars_dict["current_date"] = self.current_date

        # Return prompt (variables will be substituted when format() is called)
        return prompt

    def format_prompt(
        self,
        name: str,
        variables: Dict[str, Any] | None = None,
        inject_date: bool = True,
    ) -> ChatPromptTemplate:
        """
        Format a prompt template with variables.

        Args:
            name: Prompt template name
            variables: Variables to substitute
            inject_date: Whether to inject current date

        Returns:
            Formatted ChatPromptTemplate
        """
        return self.get_prompt(name, variables, inject_date)

    def list_prompts(self) -> list[str]:
        """
        List all available prompt templates.

        Returns:
            List of prompt names
        """
        return list(PROMPTS.keys())

    def update_date(self) -> None:
        """Update the current date (useful for long-running services)."""
        self.current_date = datetime.now().strftime("%Y-%m-%d")

