# Xynenyx LLM Service

LangChain-based LLM service for Xynenyx with multi-provider support (OpenAI primary, extensible for Anthropic/Gemini).

## Features

- **Multi-Provider Abstraction**: OpenAI implementation with extensible architecture for future providers
- **Streaming Support**: Server-Sent Events (SSE) for real-time token streaming
- **Embedding Generation**: OpenAI text-embedding-ada-002 for RAG
- **Usage Tracking**: Automatic cost tracking to Supabase
- **Prompt Management**: Domain-specific prompts for startup/VC intelligence
- **Production-Ready**: Error handling, validation, health checks

## API Endpoints

### Health & Readiness

- `GET /health` - Health check
- `GET /ready` - Readiness check

### Completions

- `POST /complete` - Synchronous completion
- `POST /complete/stream` - Streaming completion (SSE)

### Embeddings

- `POST /embeddings` - Generate embeddings

### Providers

- `GET /providers` - List all providers
- `GET /providers/{id}` - Get provider details

## Usage Examples

### Synchronous Completion

```bash
curl -X POST http://localhost:8003/complete \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-123" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7
  }'
```

### Streaming Completion

```bash
curl -X POST http://localhost:8003/complete/stream \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-123" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me about AI startups"}
    ],
    "provider": "openai"
  }'
```

### Generate Embeddings

```bash
curl -X POST http://localhost:8003/embeddings \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-123" \
  -d '{
    "text": "Startup funding round",
    "provider": "openai"
  }'
```

### List Providers

```bash
curl http://localhost:8003/providers
```

## Environment Variables

Required environment variables (see `.env.example`):

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_ENABLED` - Enable OpenAI provider (default: true)
- `ANTHROPIC_API_KEY` - Anthropic API key (optional, for future)
- `GOOGLE_API_KEY` - Google API key (optional, for future)

## Development

### Setup

```bash
# Install dependencies
poetry install

# Copy environment file
cp .env.example .env
# Edit .env with your credentials
```

### Run Locally

```bash
# Start service
poetry run uvicorn app.main:app --port 8003 --reload
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=term-missing

# Run specific test file
poetry run pytest tests/test_api.py
```

### Docker

```bash
# Build image
docker build -t xynenyx-llm .

# Run container
docker run -p 8003:8003 --env-file .env xynenyx-llm
```

## Architecture

### Provider Abstraction

The service uses a provider abstraction pattern:

- `BaseProvider` - Abstract interface for all providers
- `OpenAIProvider` - Full OpenAI implementation using LangChain
- `ProviderRouter` - Routes requests to appropriate provider

### Prompt Templates

Domain-specific prompts for startup/VC intelligence:

- `chat_agent` - Main agent system prompt
- `rag_qa` - RAG question answering
- `intent_classification` - Intent detection
- `comparison` - Entity comparison
- `trend_analysis` - Trend analysis

### Usage Tracking

All LLM usage is tracked to Supabase `llm_usage` table with:
- User ID (from `X-User-ID` header)
- Provider and model
- Token usage (prompt, completion, total)
- Cost calculation (USD)
- Metadata (JSONB)

## Testing Criteria

### Service Health

```bash
curl http://localhost:8003/health
# Expected: {"status": "healthy"}
```

### Completion Test

```bash
curl -X POST http://localhost:8003/complete \
  -H "Content-Type: application/json" \
  -H "X-User-ID: test-user" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "provider": "openai"
  }'
```

### Embedding Test

```bash
curl -X POST http://localhost:8003/embeddings \
  -H "Content-Type: application/json" \
  -H "X-User-ID: test-user" \
  -d '{
    "text": "Startup funding round",
    "provider": "openai"
  }'
```

### Unit Tests

```bash
poetry run pytest -v --cov=app --cov-report=term-missing
# Expected: >80% coverage, all tests pass
```

## Future Enhancements

- Anthropic provider implementation
- Google Gemini provider implementation
- Provider fallback logic
- Circuit breaker pattern
- Rate limiting per user
- Prompt versioning system
