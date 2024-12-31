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

The application can be configured through environment variables or a `.env` file:

### Provider Configuration

Provider-specific settings can be configured using the `PROVIDER_CONFIGS` environment variable:
