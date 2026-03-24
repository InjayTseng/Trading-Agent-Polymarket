# LLM Clients - Consistency Improvements

## Issues to Fix

### ~~1. `validate_model()` is never called~~ ✅ FIXED
- `_validate_client()` in `factory.py` calls `client.validate_model()` after every client creation
- Issues warning (not error) for unknown models

### ~~2. Inconsistent parameter handling~~ ✅ FIXED
- `GoogleClient` now maps `api_key` → `google_api_key` so callers can use unified `api_key`
- All clients accept `api_key` consistently

### ~~3. `base_url` accepted but ignored~~ ✅ FIXED
- `AnthropicClient`: now passes `base_url` as `anthropic_api_url` for custom endpoints/proxies
- `GoogleClient`: `base_url` kept in base class but intentionally unused (Google doesn't support custom endpoints)

### 4. Update validators.py with models from CLI
- Sync `VALID_MODELS` dict with CLI model options after Feature 2 is complete
