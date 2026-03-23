# TradingAgents/prediction_market/graph/setup.py

from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.prediction_market.agents import *
from tradingagents.prediction_market.agents.utils.pm_agent_states import PMAgentState

from .conditional_logic import PMConditionalLogic


class PMGraphSetup:
    """Handles the setup and configuration of the prediction market agent graph."""

    def __init__(
        self,
        quick_thinking_llm: BaseChatModel,
        deep_thinking_llm: BaseChatModel,
        tool_nodes: Dict[str, ToolNode],
        yes_memory,
        no_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: PMConditionalLogic,
        config: Dict[str, Any] = None,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.yes_memory = yes_memory
        self.no_memory = no_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.config = config or {}

    def setup_graph(
        self, selected_analysts=None,
    ):
        """Set up and compile the prediction market agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "event": Event analyst
                - "odds": Odds analyst
                - "information": Information analyst
                - "sentiment": Sentiment analyst
        """
        if selected_analysts is None:
            selected_analysts = ["event", "odds", "information", "sentiment"]
        if len(selected_analysts) == 0:
            raise ValueError("PM Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "event" in selected_analysts:
            analyst_nodes["event"] = create_event_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["event"] = create_msg_delete()
            tool_nodes["event"] = self.tool_nodes["event"]

        if "odds" in selected_analysts:
            analyst_nodes["odds"] = create_odds_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["odds"] = create_msg_delete()
            tool_nodes["odds"] = self.tool_nodes["odds"]

        if "information" in selected_analysts:
            analyst_nodes["information"] = create_information_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["information"] = create_msg_delete()
            tool_nodes["information"] = self.tool_nodes["information"]

        if "sentiment" in selected_analysts:
            analyst_nodes["sentiment"] = create_sentiment_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["sentiment"] = create_msg_delete()
            tool_nodes["sentiment"] = self.tool_nodes["sentiment"]

        # Create researcher and manager nodes
        yes_researcher_node = create_yes_researcher(
            self.quick_thinking_llm, self.yes_memory
        )
        no_researcher_node = create_no_researcher(
            self.quick_thinking_llm, self.no_memory
        )
        research_manager_node = create_pm_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_pm_trader(self.quick_thinking_llm, self.trader_memory, config=self.config)

        # Create risk analysis nodes
        aggressive_analyst = create_pm_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_pm_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_pm_conservative_debator(self.quick_thinking_llm)
        risk_manager_node = create_pm_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory
        )

        # Create workflow
        workflow = StateGraph(PMAgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("YES Researcher", yes_researcher_node)
        workflow.add_node("NO Researcher", no_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Define edges — analysts run in PARALLEL from START
        # Each analyst has its own tool loop, and all converge at YES Researcher
        for analyst_type in selected_analysts:
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # All analysts start in parallel from START
            workflow.add_edge(START, current_analyst)

            # Add conditional edges for tool loop
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # All analysts converge at YES Researcher
            workflow.add_edge(current_clear, "YES Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "YES Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "NO Researcher": "NO Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "NO Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "YES Researcher": "YES Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        # Skip risk debate when Trader recommends PASS (saves 4-5 LLM calls)
        workflow.add_conditional_edges(
            "Trader",
            self.conditional_logic.should_skip_risk_debate,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "END": END,
            },
        )
        risk_routing = {
            "Aggressive Analyst": "Aggressive Analyst",
            "Conservative Analyst": "Conservative Analyst",
            "Neutral Analyst": "Neutral Analyst",
            "Risk Judge": "Risk Judge",
        }
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            risk_routing,
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            risk_routing,
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            risk_routing,
        )

        workflow.add_edge("Risk Judge", END)

        # Compile and return
        return workflow.compile()
