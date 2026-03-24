# TradingAgents/prediction_market/graph/conditional_logic.py

import logging
from tradingagents.prediction_market.agents.utils.pm_agent_states import PMAgentState

logger = logging.getLogger(__name__)


class PMConditionalLogic:
    """Handles conditional logic for determining prediction market graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1, max_tool_calls=10):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds
        self.max_tool_calls = max_tool_calls

    def _count_tool_calls(self, state: PMAgentState) -> int:
        """Count total tool call messages in the current conversation."""
        return sum(1 for m in state["messages"] if getattr(m, "tool_calls", None))

    def _should_continue_tool_loop(self, state: PMAgentState, tool_node: str, done_node: str) -> str:
        """Generic tool loop continuation with safety limit."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            tool_call_count = self._count_tool_calls(state)
            if tool_call_count >= self.max_tool_calls:
                logger.warning(
                    "Tool call limit (%d) reached for analyst, forcing completion",
                    self.max_tool_calls,
                )
                return done_node
            return tool_node
        return done_node

    def should_continue_event(self, state: PMAgentState):
        """Determine if event analysis should continue."""
        return self._should_continue_tool_loop(state, "tools_event", "Msg Clear Event")

    def should_continue_odds(self, state: PMAgentState):
        """Determine if odds analysis should continue."""
        return self._should_continue_tool_loop(state, "tools_odds", "Msg Clear Odds")

    def should_continue_information(self, state: PMAgentState):
        """Determine if information analysis should continue."""
        return self._should_continue_tool_loop(state, "tools_information", "Msg Clear Information")

    def should_continue_sentiment(self, state: PMAgentState):
        """Determine if sentiment analysis should continue."""
        return self._should_continue_tool_loop(state, "tools_sentiment", "Msg Clear Sentiment")

    def should_continue_debate(self, state: PMAgentState) -> str:
        """Determine if YES/NO debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("YES"):
            return "NO Researcher"
        return "YES Researcher"

    def should_skip_risk_debate(self, state: PMAgentState) -> str:
        """Skip the risk debate entirely if the Trader recommends PASS."""
        trader_plan = state.get("trader_investment_plan", "")
        if trader_plan:
            upper_plan = trader_plan.upper()
            # Check if the trader's final recommendation is PASS
            if "FINAL TRADE PROPOSAL: **PASS**" in upper_plan or (
                "PASS" in upper_plan and "BUY_YES" not in upper_plan and "BUY_NO" not in upper_plan
            ):
                return "END"
        return "Aggressive Analyst"

    def should_continue_risk_analysis(self, state: PMAgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # rounds of back-and-forth between 3 agents
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
