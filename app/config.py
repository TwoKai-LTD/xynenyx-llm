"""Configuration settings for Xynenyx LLM Service."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import Dict, Any


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service settings
    app_name: str = "Xynenyx LLM Service"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8003

    # Supabase settings
    supabase_url: str
    supabase_service_role_key: str | None = None
    # Support alternative name from .env (SUPABASE_SERVICE_KEY)
    supabase_service_key: str | None = None

    # OpenAI settings
    openai_api_key: str

    # Anthropic settings (optional, for future)
    anthropic_api_key: str | None = None

    # Google settings (optional, for future)
    google_api_key: str | None = None

    # Provider configuration
    openai_enabled: bool = True
    anthropic_enabled: bool = False
    google_enabled: bool = False

    # Model defaults
    default_model: str = "gpt-4o-mini"
    default_embedding_model: str = "text-embedding-ada-002"

    # Timeout settings (seconds)
    request_timeout: int = 60
    streaming_timeout: int = 300

    # Cost rates per 1K tokens (input, output)
    cost_rates: Dict[str, Dict[str, float]] = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    }

    # CORS settings
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    @model_validator(mode="after")
    def resolve_service_role_key(self):
        """Resolve service role key from either variable name."""
        if not self.supabase_service_role_key and self.supabase_service_key:
            self.supabase_service_role_key = self.supabase_service_key
        if not self.supabase_service_role_key:
            raise ValueError("Either supabase_service_role_key or supabase_service_key must be set")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )


settings = Settings()

