from langchain_core.messages import HumanMessage, RemoveMessage

from tradingagents.prediction_market.agents.utils.pm_tools import (
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


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility."""
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        placeholder = HumanMessage(content="Continue")
        return {"messages": removal_operations + [placeholder]}

    return delete_messages
