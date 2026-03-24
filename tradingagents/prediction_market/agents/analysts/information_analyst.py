from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.prediction_market.agents.utils.pm_agent_utils import (
    get_news,
    get_global_news,
    get_related_markets,
    get_crypto_data,
    get_crypto_history,
    get_defi_protocol_tvl,
    get_top_defi,
)


def create_information_analyst(llm):
    def information_analyst_node(state):
        current_date = state["trade_date"]
        market_id = state["market_id"]
        market_question = state["market_question"]

        tools = [
            get_news,
            get_global_news,
            get_related_markets,
            get_crypto_data,
            get_crypto_history,
            get_defi_protocol_tvl,
            get_top_defi,
        ]

        system_message = (
            "You are an Information Analyst for prediction markets. Your task is to find and analyze news, "
            "data, and developments that are relevant to the outcome of the prediction market event. "
            "Use the available tools to search for news and related markets. Your analysis should cover:\n"
            "IMPORTANT: Before analyzing specific evidence, first establish the HISTORICAL BASE RATE for this class of event. "
            "For example: What percentage of similar events have occurred historically? (e.g., incumbent reelection rates, "
            "merger completion rates, policy passage rates). State the base rate and your source explicitly.\n\n"
            "Your analysis should cover:\n"
            "1. Historical base rate for this class of event (your probabilistic anchor)\n"
            "2. Recent news and developments directly related to the event being predicted\n"
            "3. Broader macro or contextual factors that could influence the outcome\n"
            "4. Information the market may not have priced in yet (information edge)\n"
            "5. Assessment of how new information shifts the probability from the base rate\n"
            "6. Related markets and what their prices signal about this event\n"
            "7. Key upcoming catalysts or data releases that could move the market\n"
            "Do not simply state that the information is mixed, provide detailed and finegrained analysis "
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
                    "When calling get_news, focus on factual developments: policy decisions, data releases, "
                    "institutional actions, legal rulings. Leave opinion and sentiment analysis to the Sentiment Analyst.\n"
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
            update["information_report"] = result.content
        return update

    return information_analyst_node
