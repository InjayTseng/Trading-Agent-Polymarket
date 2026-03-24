# TradingAgents/prediction_market/graph/pm_trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.prediction_market.agents import *
from tradingagents.prediction_market.pm_config import PM_DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.prediction_market.agents.utils.pm_agent_states import (
    PMAgentState,
    PMInvestDebateState,
    PMRiskDebateState,
)

# Import PM tool functions
from tradingagents.prediction_market.agents.utils.pm_agent_utils import (
    get_market_info,
    get_market_price_history,
    get_order_book,
    get_resolution_criteria,
    get_event_context,
    get_related_markets,
    search_markets,
    get_news,
    get_global_news,
    get_crypto_data,
    get_crypto_history,
    get_defi_protocol_tvl,
    get_top_defi,
)

from .conditional_logic import PMConditionalLogic
from .setup import PMGraphSetup
from .propagation import PMPropagator
from .reflection import PMReflector
from .signal_processing import PMSignalProcessor


class PMTradingAgentsGraph:
    """Main class that orchestrates the prediction market trading agents framework."""

    VALID_ANALYSTS = {"event", "odds", "information", "sentiment"}

    def __init__(
        self,
        selected_analysts=None,
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the prediction market trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses PM default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        if selected_analysts is None:
            selected_analysts = ["event", "odds", "information", "sentiment"]
        invalid = set(selected_analysts) - self.VALID_ANALYSTS
        if invalid:
            raise ValueError(f"Unknown analyst types: {invalid}. Valid: {self.VALID_ANALYSTS}")
        if not selected_analysts:
            raise ValueError("At least one analyst must be selected.")

        self.debug = debug
        self.config = config or PM_DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Validate required config keys
        required_keys = {"llm_provider", "deep_think_llm", "quick_think_llm",
                         "max_debate_rounds", "max_risk_discuss_rounds", "project_dir"}
        missing = required_keys - self.config.keys()
        if missing:
            raise ValueError(f"Config is missing required keys: {missing}")

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        # Initialize memories
        self.yes_memory = FinancialSituationMemory("yes_memory", self.config)
        self.no_memory = FinancialSituationMemory("no_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = PMConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
            max_tool_calls=self.config.get("max_tool_calls", 10),
        )
        self.graph_setup = PMGraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.yes_memory,
            self.no_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            config=self.config,
        )

        self.propagator = PMPropagator(max_recur_limit=self.config.get("max_recur_limit", 100))
        self.reflector = PMReflector(self.quick_thinking_llm)
        self.signal_processor = PMSignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.market_id = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different prediction market data sources."""
        return {
            "event": ToolNode(
                [
                    # Event context and resolution
                    get_market_info,
                    get_resolution_criteria,
                    get_event_context,
                ]
            ),
            "odds": ToolNode(
                [
                    # Price, order book, and market data
                    get_market_info,
                    get_market_price_history,
                    get_order_book,
                ]
            ),
            "information": ToolNode(
                [
                    # News and related markets
                    get_news,
                    get_global_news,
                    get_related_markets,
                    search_markets,
                    # On-chain / crypto data
                    get_crypto_data,
                    get_crypto_history,
                    get_defi_protocol_tvl,
                    get_top_defi,
                ]
            ),
            "sentiment": ToolNode(
                [
                    # News and market search for sentiment analysis
                    get_news,
                    get_global_news,
                    search_markets,
                ]
            ),
        }

    def propagate(self, market_id, trade_date, market_question=""):
        """Run the prediction market trading agents graph for a market on a specific date.

        Args:
            market_id: The Polymarket condition ID or market identifier
            trade_date: The date of analysis
            market_question: Optional full text of the market question
        """

        self.market_id = market_id

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            market_id, trade_date, market_question
        )

        if self.debug:
            # Debug mode with tracing
            args = self.propagator.get_graph_args(stream=True)
            final_state = None
            for chunk in self.graph.stream(init_agent_state, **args):
                messages = chunk.get("messages", [])
                if messages:
                    messages[-1].pretty_print()
                final_state = chunk

            if final_state is None:
                raise RuntimeError("Graph produced no output chunks.")
        else:
            # Standard mode without tracing
            args = self.propagator.get_graph_args(stream=False)
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        # If risk debate was skipped (PASS fast-path), use trader plan as decision
        raw_decision = final_state.get("final_trade_decision", "")
        if not raw_decision:
            trader_plan = final_state.get("trader_investment_plan", "")
            if trader_plan:
                raw_decision = trader_plan
            else:
                import warnings
                warnings.warn(f"Graph did not produce a final_trade_decision for market {self.market_id}")
        return final_state, self.process_signal(raw_decision or "")

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        invest_debate = final_state.get("investment_debate_state", {})
        risk_debate = final_state.get("risk_debate_state", {})

        self.log_states_dict[str(trade_date)] = {
            "market_id": final_state.get("market_id", ""),
            "market_question": final_state.get("market_question", ""),
            "trade_date": final_state.get("trade_date", ""),
            "event_report": final_state.get("event_report", ""),
            "odds_report": final_state.get("odds_report", ""),
            "information_report": final_state.get("information_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "investment_debate_state": {
                "yes_history": invest_debate.get("yes_history", ""),
                "no_history": invest_debate.get("no_history", ""),
                "history": invest_debate.get("history", ""),
                "current_response": invest_debate.get("current_response", ""),
                "judge_decision": invest_debate.get("judge_decision", ""),
            },
            "trader_investment_plan": final_state.get("trader_investment_plan", ""),
            "risk_debate_state": {
                "aggressive_history": risk_debate.get("aggressive_history", ""),
                "conservative_history": risk_debate.get("conservative_history", ""),
                "neutral_history": risk_debate.get("neutral_history", ""),
                "history": risk_debate.get("history", ""),
                "judge_decision": risk_debate.get("judge_decision", ""),
            },
            "investment_plan": final_state.get("investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        }

        # Save to file using config results_dir
        results_dir = Path(os.path.abspath(self.config.get("results_dir", "./results")))
        log_dir = results_dir / str(self.market_id) / "PMTradingAgentsStrategy_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"full_states_log_{trade_date}.json"

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        if self.curr_state is None:
            raise RuntimeError("reflect_and_remember() called before a successful propagate() run.")
        self.reflector.reflect_yes_researcher(
            self.curr_state, returns_losses, self.yes_memory
        )
        self.reflector.reflect_no_researcher(
            self.curr_state, returns_losses, self.no_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
