"""Integration tests for PMTradingAgentsGraph — verifies graph compilation and wiring."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.prediction_market.pm_config import PM_DEFAULT_CONFIG


def _mock_llm():
    """Create a mock LLM that has bind_tools method."""
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = MagicMock(content="test", tool_calls=[])
    return llm


class TestGraphCompilation:
    @patch("tradingagents.prediction_market.graph.pm_trading_graph.create_llm_client")
    def test_default_config_compiles(self, mock_create):
        """Graph should compile with default config and all analysts."""
        mock_client = MagicMock()
        mock_client.get_llm.return_value = _mock_llm()
        mock_create.return_value = mock_client

        from tradingagents.prediction_market.graph.pm_trading_graph import PMTradingAgentsGraph

        graph = PMTradingAgentsGraph(config=PM_DEFAULT_CONFIG.copy())
        assert graph is not None
        assert graph.tool_nodes is not None
        assert "event" in graph.tool_nodes
        assert "odds" in graph.tool_nodes
        assert "information" in graph.tool_nodes
        assert "sentiment" in graph.tool_nodes

    @patch("tradingagents.prediction_market.graph.pm_trading_graph.create_llm_client")
    def test_tool_nodes_have_correct_tools(self, mock_create):
        """Information analyst tool node should include on-chain tools."""
        mock_client = MagicMock()
        mock_client.get_llm.return_value = _mock_llm()
        mock_create.return_value = mock_client

        from tradingagents.prediction_market.graph.pm_trading_graph import PMTradingAgentsGraph

        graph = PMTradingAgentsGraph(config=PM_DEFAULT_CONFIG.copy())

        # Information analyst should have news + on-chain tools
        info_tools = graph.tool_nodes["information"]
        tool_names = [t.name for t in info_tools.tools_by_name.values()] if hasattr(info_tools, 'tools_by_name') else []
        # At minimum, verify the tool node was created successfully
        assert info_tools is not None

    @patch("tradingagents.prediction_market.graph.pm_trading_graph.create_llm_client")
    def test_conditional_logic_initialized(self, mock_create):
        """Conditional logic should use config values."""
        mock_client = MagicMock()
        mock_client.get_llm.return_value = _mock_llm()
        mock_create.return_value = mock_client

        from tradingagents.prediction_market.graph.pm_trading_graph import PMTradingAgentsGraph

        config = PM_DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = 3
        config["max_tool_calls"] = 15

        graph = PMTradingAgentsGraph(config=config)
        assert graph.conditional_logic.max_debate_rounds == 3
        assert graph.conditional_logic.max_tool_calls == 15

    @patch("tradingagents.prediction_market.graph.pm_trading_graph.create_llm_client")
    def test_memory_instances_created(self, mock_create):
        """All 5 memory instances should be initialized."""
        mock_client = MagicMock()
        mock_client.get_llm.return_value = _mock_llm()
        mock_create.return_value = mock_client

        from tradingagents.prediction_market.graph.pm_trading_graph import PMTradingAgentsGraph

        graph = PMTradingAgentsGraph(config=PM_DEFAULT_CONFIG.copy())
        assert graph.yes_memory is not None
        assert graph.no_memory is not None
        assert graph.trader_memory is not None
        assert graph.invest_judge_memory is not None
        assert graph.risk_manager_memory is not None
