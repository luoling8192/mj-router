# AI Image Generation API

A unified FastAPI service for generating images through multiple AI providers (DALL-E, Midjourney). Integrates with [midjourney-proxy](https://github.com/novicezk/midjourney-proxy) for Midjourney support.

## Key Features

🎨 Core Features:
- 🔄 Asynchronous image generation with real-time progress tracking
- 🤖 Support for multiple AI providers (DALL-E, Midjourney)
- 📚 Comprehensive API documentation with OpenAPI/Swagger

⚙️ Technical Features:
- 🔌 Provider-agnostic interface with extensible provider system
- 🛡️ Robust error handling and retry mechanisms
- ⏳ Rate limiting and request queueing
- 📊 Structured logging and monitoring

## Getting Started

### System Requirements

- Python 3.12 or higher
- Package manager: uv
- For Midjourney integration:
  - Running midjourney-proxy instance
  - Discord account with active Midjourney subscription
  - Discord server and channel configuration

### Setup Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/luoling8192/mj-router.git
   cd mj-router
   ```

2. Set up Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: `.venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Launch development server:
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```
