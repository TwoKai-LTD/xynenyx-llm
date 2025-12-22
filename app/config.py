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
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or JSON list."""
        if isinstance(v, str):
            # Try JSON first
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                # If not JSON, treat as comma-separated string
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def validate_config(self):
        """Validate all required configuration."""
        errors = []
        
        # Resolve service role key
        if not self.supabase_service_role_key and self.supabase_service_key:
            self.supabase_service_role_key = self.supabase_service_key
        
        # Validate Supabase
        if not self.supabase_url:
            errors.append("SUPABASE_URL is required")
        elif not self.supabase_url.startswith("http"):
            errors.append("SUPABASE_URL must be a valid HTTP/HTTPS URL")
        
        if not self.supabase_service_role_key:
            errors.append("Either SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY must be set")
        
        # Validate OpenAI if enabled
        if self.openai_enabled:
            if not self.openai_api_key:
                errors.append("OPENAI_API_KEY is required when OPENAI_ENABLED=true")
            elif self.openai_api_key.startswith("sk-your") or len(self.openai_api_key) < 20:
                errors.append("OPENAI_API_KEY appears to be invalid or placeholder")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )


settings = Settings()

