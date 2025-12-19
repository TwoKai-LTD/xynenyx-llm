# Xynenyx LLM Service

LangChain LLM service providing OpenAI integration with extensible provider abstraction, prompt management, and usage tracking.

## Overview

The LLM service:

- Implements OpenAI provider (primary)
- Uses provider abstraction pattern (extensible for future providers)
- Implements smart routing and fallback
- Manages domain-specific prompt templates
- Tracks usage and costs
- Provides embedding generation

## Quick Start

### Local Development

```bash
# Install dependencies
poetry install

# Run locally
poetry run uvicorn app.main:app --port 8003 --reload
```

### Docker

```bash
docker build -t xynenyx-llm .
docker run -p 8003:8003 --env-file .env xynenyx-llm
```

## API Endpoints

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /complete` - Synchronous completion
- `POST /complete/stream` - Streaming completion (SSE)
- `POST /embeddings` - Generate embeddings
- `GET /providers` - List available providers
- `GET /providers/{id}` - Provider details

## Configuration

See `.env.example` for all configuration options.

## Testing

```bash
poetry run pytest -v
poetry run pytest --cov=app --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) file

