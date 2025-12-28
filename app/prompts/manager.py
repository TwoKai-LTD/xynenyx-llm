"""Prompt manager with versioning support."""
import logging
from typing import Dict, Optional
from app.prompts.templates import PROMPTS, get_prompt

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates with versioning support."""

    def __init__(self):
        """Initialize prompt manager."""
        self.versions: Dict[str, Dict[str, str]] = {}
        self.current_versions: Dict[str, str] = {}
        self.metrics: Dict[str, Dict[str, any]] = {}

    def register_version(
        self,
        prompt_name: str,
        version: str,
        prompt_content: str,
        set_as_current: bool = False,
    ) -> None:
        """
        Register a new version of a prompt.

        Args:
            prompt_name: Name of the prompt (e.g., "rag_qa")
            version: Version identifier (e.g., "v1", "v2", "2025-01-15")
            prompt_content: The prompt template content
            set_as_current: Whether to set this as the current version
        """
        if prompt_name not in self.versions:
            self.versions[prompt_name] = {}

        self.versions[prompt_name][version] = prompt_content

        if set_as_current:
            self.current_versions[prompt_name] = version

        logger.info(f"Registered prompt version: {prompt_name}@{version}")

    def get_prompt(
        self,
        prompt_name: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Get a prompt template by name and version.

        Args:
            prompt_name: Name of the prompt
            version: Optional version identifier (defaults to current version)

        Returns:
            Prompt template content
        """
        # If version specified, use it
        if version:
            if prompt_name in self.versions and version in self.versions[prompt_name]:
                return self.versions[prompt_name][version]
            logger.warning(f"Version {version} not found for {prompt_name}, using default")

        # Use current version if set
        if prompt_name in self.current_versions:
            current_version = self.current_versions[prompt_name]
            if prompt_name in self.versions and current_version in self.versions[prompt_name]:
                return self.versions[prompt_name][current_version]

        # Fallback to default prompt from templates
        try:
            prompt_template = get_prompt(prompt_name)
            # Extract system message content
            messages = prompt_template.messages
            for role, content in messages:
                if role == "system":
                    return content
            return str(prompt_template)
        except ValueError:
            logger.error(f"Prompt {prompt_name} not found")
            return ""

    def set_current_version(self, prompt_name: str, version: str) -> None:
        """
        Set the current version for a prompt.

        Args:
            prompt_name: Name of the prompt
            version: Version identifier to set as current
        """
        if prompt_name not in self.versions:
            logger.warning(f"Prompt {prompt_name} has no registered versions")
            return

        if version not in self.versions[prompt_name]:
            logger.warning(f"Version {version} not found for {prompt_name}")
            return

        self.current_versions[prompt_name] = version
        logger.info(f"Set current version for {prompt_name} to {version}")

    def track_metric(
        self,
        prompt_name: str,
        version: str,
        metric_name: str,
        value: float,
    ) -> None:
        """
        Track a performance metric for a prompt version.

        Args:
            prompt_name: Name of the prompt
            version: Version identifier
            metric_name: Name of the metric (e.g., "accuracy", "latency", "user_satisfaction")
            value: Metric value
        """
        key = f"{prompt_name}@{version}"
        if key not in self.metrics:
            self.metrics[key] = {}

        if metric_name not in self.metrics[key]:
            self.metrics[key][metric_name] = []

        self.metrics[key][metric_name].append(value)

    def get_metrics(
        self,
        prompt_name: str,
        version: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Get metrics for a prompt version.

        Args:
            prompt_name: Name of the prompt
            version: Optional version identifier (defaults to current version)

        Returns:
            Dictionary of metrics
        """
        if version:
            key = f"{prompt_name}@{version}"
        else:
            current_version = self.current_versions.get(prompt_name, "default")
            key = f"{prompt_name}@{current_version}"

        return self.metrics.get(key, {})

    def list_versions(self, prompt_name: str) -> list[str]:
        """
        List all versions for a prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            List of version identifiers
        """
        if prompt_name not in self.versions:
            return []
        return list(self.versions[prompt_name].keys())


# Global prompt manager instance
_prompt_manager = PromptManager()


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    return _prompt_manager
