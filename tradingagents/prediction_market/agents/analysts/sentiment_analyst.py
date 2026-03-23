from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.prediction_market.agents.utils.pm_agent_utils import (
    get_news,
    get_global_news,
    search_markets,
)


def create_sentiment_analyst(llm):
    def sentiment_analyst_node(state):
        current_date = state["trade_date"]
        market_id = state["market_id"]
        market_question = state["market_question"]

        tools = [
            get_news,
            get_global_news,
            search_markets,
        ]

        system_message = (
            "You are a Sentiment Analyst for prediction markets. Your task is to analyze news narrative tone, "
            "expert commentary, and crowd sentiment around the prediction market event. "
            "Use the available tools to search for news sentiment and related market activity. "
            "Important: You do not have access to social media APIs or polling data directly. "
            "Infer sentiment from news article tone, language, and framing. "
            "Note explicitly when data is absent rather than inferring it.\n"
            "Your analysis should cover:\n"
            "1. News narrative tone and framing around the event\n"
            "2. Expert commentary and analyst forecasts found in news articles\n"
            "3. Expert vs crowd divergence - where do domain experts disagree with market prices?\n"
            "4. Narrative momentum - is sentiment shifting in a particular direction?\n"
            "5. Sentiment extremes that may signal contrarian opportunities\n"
            "6. Related market sentiment and cross-market signals\n"
            "When calling get_news, focus on opinion, commentary, and narrative framing — not raw factual reporting. "
            "Search for terms like 'opinion', 'analysts say', 'experts predict', 'poll shows' to surface sentiment-bearing content.\n"
            "Do not simply state that the sentiment is mixed, provide detailed and finegrained analysis "
            "and insights that may help traders make decisions."
            """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_message}\n"
                    "You have access to the following tools: {tool_names}.\n"
                    "Use the provided tools to gather data before writing your report.\n"
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
            update["sentiment_report"] = result.content
        return update

    return sentiment_analyst_node
