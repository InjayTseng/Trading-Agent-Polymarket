import os

PM_DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.path.abspath(os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results")),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.2",
    "quick_think_llm": "gpt-5-mini",
    "backend_url": "https://api.openai.com/v1",
    # Provider-specific thinking configuration
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    # Polymarket API
    "polymarket_gamma_url": "https://gamma-api.polymarket.com",
    "polymarket_clob_url": "https://clob.polymarket.com",
    # Trading parameters
    "kelly_fraction": 0.25,
    "min_edge_threshold": 0.05,
    "max_position_pct": 0.05,
    "max_cluster_exposure_pct": 0.15,
    "bankroll": 10000,
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool loop safety limit (per analyst)
    "max_tool_calls": 10,
    # Signal processing
    "max_position_size_cap": 0.10,
    # Cache TTLs (seconds)
    "cache_ttl_gamma": 300,
    "cache_ttl_clob_book": 30,
    "cache_ttl_clob_prices": 300,
    "cache_ttl_events": 600,
    "cache_ttl_markets": 300,
    # WebSocket settings
    "ws_market_url": "wss://ws-subscriptions-clob.polymarket.com/ws/market",
    "ws_ping_interval": 10,
    "ws_max_assets_per_connection": 500,
}
