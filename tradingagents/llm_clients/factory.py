import warnings
from typing import Optional

from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .google_client import GoogleClient


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Args:
        provider: LLM provider (openai, anthropic, google, xai, ollama, openrouter)
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments
            - http_client: Custom httpx.Client for SSL proxy or certificate customization
            - http_async_client: Custom httpx.AsyncClient for async operations
            - timeout: Request timeout in seconds
            - max_retries: Maximum retry attempts
            - api_key: API key for the provider
            - callbacks: LangChain callbacks

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    if provider_lower in ("openai", "ollama", "openrouter"):
        client = OpenAIClient(model, base_url, provider=provider_lower, **kwargs)
        _validate_client(client, provider, model)
        return client

    if provider_lower == "xai":
        client = OpenAIClient(model, base_url, provider="xai", **kwargs)
        _validate_client(client, provider, model)
        return client

    if provider_lower == "anthropic":
        client = AnthropicClient(model, base_url, **kwargs)
        _validate_client(client, provider, model)
        return client

    if provider_lower == "google":
        client = GoogleClient(model, base_url, **kwargs)
        _validate_client(client, provider, model)
        return client

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _validate_client(client: BaseLLMClient, provider: str, model: str):
    """Validate the model after client creation."""
    if not client.validate_model():
        warnings.warn(
            f"Model '{model}' may not be valid for provider '{provider}'. "
            f"Proceeding anyway — the API will reject it if invalid.",
            stacklevel=3,
        )
