"""Provider management endpoints."""
from fastapi import APIRouter, HTTPException

from app.schemas.providers import ProviderInfo, ProviderListResponse
from app.providers.router import ProviderRouter

router = APIRouter(prefix="/providers", tags=["providers"])

# Initialize dependencies
provider_router = ProviderRouter()


@router.get("", response_model=ProviderListResponse)
async def list_providers():
    """List all available providers with their status."""
    providers_dict = provider_router.list_providers()
    health_status = await provider_router.check_all_health()

    providers = []
    for provider_id, info in providers_dict.items():
        providers.append(
            ProviderInfo(
                id=provider_id,
                name=info["name"],
                enabled=info["enabled"],
                healthy=health_status.get(provider_id, False),
                models=info["models"],
                metadata={},
            )
        )

    return ProviderListResponse(providers=providers)


@router.get("/{provider_id}", response_model=ProviderInfo)
async def get_provider(provider_id: str):
    """Get details for a specific provider."""
    try:
        provider = provider_router.get_provider(provider_id)
        health_status = await provider_router.check_provider_health(provider_id)

        return ProviderInfo(
            id=provider_id,
            name=provider.name,
            enabled=True,
            healthy=health_status,
            models=provider.get_supported_models(),
            metadata={},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

