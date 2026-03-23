from tradingagents.prediction_market import PMTradingAgentsGraph
from tradingagents.prediction_market.pm_config import PM_DEFAULT_CONFIG
from dotenv import load_dotenv

load_dotenv()

config = PM_DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-sonnet-4-6"
config["quick_think_llm"] = "claude-sonnet-4-6"

pm = PMTradingAgentsGraph(debug=True, config=config)

# Example: Byron Donalds 2028 Republican nomination
_, decision = pm.propagate("561986", "2026-03-23", "Will Byron Donalds win the 2028 Republican presidential nomination?")
print(decision)
