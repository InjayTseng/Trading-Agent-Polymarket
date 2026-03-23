from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.prediction_market.agents.utils.pm_agent_utils import (
    get_market_info,
    get_resolution_criteria,
    get_event_context,
)


def create_event_analyst(llm):
    def event_analyst_node(state):
        current_date = state["trade_date"]
        market_id = state["market_id"]
        market_question = state["market_question"]

        tools = [
            get_market_info,
            get_resolution_criteria,
            get_event_context,
        ]

        system_message = (
            "You are an Event Analyst for prediction markets. Your task is to analyze the prediction market event itself. "
            "Understand what is being predicted, how the market resolves, and the timeline. "
            "Use the available tools to gather market info and resolution criteria. "
            "Your analysis should cover:\n"
            "1. Event description and what exactly is being predicted\n"
            "2. Resolution criteria - how will the outcome be determined? Is it clear or ambiguous?\n"
            "3. Key dates and triggers that could cause resolution\n"
            "4. Resolution ambiguity assessment (clear/moderate/ambiguous)\n"
            "5. Related markets within the same event if applicable\n"
            "6. Market structure type - is this a standard binary market or a negRisk (multi-outcome cluster)? "
            "If negRisk=True, all outcomes in the event are mutually exclusive and probabilities sum to 1.\n"
            "Do not simply state that the situation is unclear, provide detailed and finegrained analysis "
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
            update["event_report"] = result.content
        return update

    return event_analyst_node
