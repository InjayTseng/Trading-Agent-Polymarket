from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.prediction_market.agents.utils.pm_agent_utils import (
    get_market_info,
    get_market_price_history,
    get_order_book,
)


def create_odds_analyst(llm):
    def odds_analyst_node(state):
        current_date = state["trade_date"]
        market_id = state["market_id"]
        market_question = state["market_question"]

        tools = [
            get_market_info,
            get_market_price_history,
            get_order_book,
        ]

        system_message = (
            "You are an Odds Analyst for prediction markets. Your task is to analyze the market microstructure "
            "and pricing dynamics of the prediction market. Use the available tools to gather market data, "
            "price history, and order book information. Your analysis should cover:\n"
            "1. Current price/probability and what it implies about market consensus\n"
            "2. Bid-ask spread and liquidity assessment - how easy is it to enter/exit positions?\n"
            "3. Order book depth - are there large resting orders that indicate informed traders?\n"
            "4. Price history trends - has the market been trending, mean-reverting, or volatile?\n"
            "5. Market efficiency assessment - are there signs of mispricing or stale prices?\n"
            "6. Market lifecycle stage (early/mid/late) based on time to resolution and volume patterns\n"
            "Do not simply state that the trends are mixed, provide detailed and finegrained analysis "
            "and insights that may help traders make decisions."
            """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_message}\n"
                    "You have access to the following tools: {tool_names}.\n"
                    "Use the provided tools to gather data before writing your report. "
                    "You MUST call all available tools before writing your final report.\n"
                    "When fetching price history, use a lookback window appropriate to the market's remaining life:\n"
                    "  - Markets expiring in < 7 days: last 30 days\n"
                    "  - Markets expiring in 7-30 days: last 90 days\n"
                    "  - Markets expiring in > 30 days: last 180 days\n"
                    "Call get_market_info first to determine the end date before requesting price history.\n"
                    "For your reference, the current date is {current_date}. Market ID: {market_id}. Question: {market_question}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(market_id=market_id)
        prompt = prompt.partial(market_question=market_question)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        update = {"messages": [result]}
        if not result.tool_calls:
            update["odds_report"] = result.content
        return update

    return odds_analyst_node
