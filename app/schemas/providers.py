"""Pydantic schemas for provider information."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ProviderInfo(BaseModel):
    """Information about a provider."""

    id: str = Field(..., description="Provider ID")
    name: str = Field(..., description="Provider name")
    enabled: bool = Field(..., description="Whether provider is enabled")
    healthy: bool = Field(..., description="Whether provider is healthy")
    models: List[str] = Field(..., description="List of supported models")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProviderListResponse(BaseModel):
    """Response listing all providers."""

    providers: List[ProviderInfo] = Field(..., description="List of providers")

