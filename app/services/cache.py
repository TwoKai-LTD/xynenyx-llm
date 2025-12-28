"""Caching service for LLM completions."""
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CompletionCache:
    """Simple in-memory cache for LLM completions."""

    def __init__(self, ttl_seconds: int = 3600):  # 1 hour default
        """
        Initialize completion cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def _get_cache_key(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """Generate cache key from messages and temperature."""
        # Only cache deterministic requests (low temperature)
        if temperature > 0.3:
            return None  # Don't cache non-deterministic requests
        
        cache_data = {
            "messages": messages,
            "temperature": temperature,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def get(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        """
        Get completion from cache.

        Args:
            messages: List of messages
            temperature: Sampling temperature

        Returns:
            Cached completion or None if not cached or expired
        """
        cache_key = self._get_cache_key(messages, temperature)
        if not cache_key:
            return None

        entry = self.cache.get(cache_key)
        if not entry:
            return None

        # Check if expired
        if datetime.now() - entry["timestamp"] > timedelta(seconds=self.ttl_seconds):
            del self.cache[cache_key]
            return None

        return entry["completion"]

    def set(
        self,
        messages: List[Dict[str, str]],
        completion: Dict[str, Any],
        temperature: float = 0.7,
    ) -> None:
        """
        Store completion in cache.

        Args:
            messages: List of messages
            completion: Completion response to cache
            temperature: Sampling temperature
        """
        cache_key = self._get_cache_key(messages, temperature)
        if not cache_key:
            return  # Don't cache non-deterministic requests

        self.cache[cache_key] = {
            "completion": completion,
            "timestamp": datetime.now(),
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Completion cache cleared")

    def size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)

