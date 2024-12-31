# AI Image Generation API

A FastAPI service that provides a unified interface for generating images using various AI providers (DALL-E, Midjourney).

## Features

- 🚀 Asynchronous image generation
- 🔄 Multiple AI provider support
- 📊 Task status tracking
- 🔐 Provider-specific API key management
- 📝 Comprehensive API documentation

## Quick Start

### Prerequisites

- Python 3.12+
- uv

### Installation

```bash
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

### Running the application

```bash
uv run src/main.py
```

## Configuration

### Environment Variables

1. Copy the example configuration file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```bash
   # Required
   OPENAI_API_KEY=sk-your-openai-key-here
   MIDJOURNEY_API_KEY=your-midjourney-key-here
   ```

3. (Optional) Customize other settings in `.env`:
   - Application settings (APP_NAME, APP_HOST, etc.)
   - Provider configurations (timeouts, retries, etc.)
   - Global request settings

### Configuration Priority

1. Environment variables take precedence over `.env` file values
2. `.env` file values override default settings
3. Default values are used if no configuration is provided

### Provider Configuration

You can customize provider-specific settings using the `PROVIDER_CONFIGS` environment variable:
