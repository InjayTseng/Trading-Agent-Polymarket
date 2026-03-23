import functools


def create_pm_trader(llm, memory, config=None):
    trading_config = config or {}
    kelly_fraction = trading_config.get("kelly_fraction", 0.25)
    min_edge_threshold = trading_config.get("min_edge_threshold", 0.05)
    max_position_pct = trading_config.get("max_position_pct", 0.05)
    bankroll = trading_config.get("bankroll", 10000)

    def trader_node(state, name):
        market_question = state["market_question"]
        investment_plan = state["investment_plan"]
        event_report = state.get("event_report", "")
        odds_report = state.get("odds_report", "")
        information_report = state.get("information_report", "")
        sentiment_report = state.get("sentiment_report", "")

        curr_situation = f"{event_report}\n\n{odds_report}\n\n{information_report}\n\n{sentiment_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": (
                f"You are evaluating a prediction market position for the following question:\n\n"
                f"MARKET QUESTION: {market_question}\n\n"
                f"Based on a comprehensive analysis by a team of analysts, here is the investment plan "
                f"synthesized from event analysis, odds analysis, information research, and sentiment analysis. "
                f"Use this plan as a foundation for your trading decision.\n\n"
                f"Proposed Investment Plan:\n{investment_plan}\n\n"
                f"Event Analysis Report:\n{event_report}\n\n"
                f"Odds Analysis Report:\n{odds_report}\n\n"
                f"Information Analysis Report:\n{information_report}\n\n"
                f"Sentiment Analysis Report:\n{sentiment_report}\n\n"
                f"Leverage these insights to make an informed and strategic trading decision."
            ),
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a prediction market trader analyzing market data to make trading decisions on binary outcome markets. Your goal is to identify mispriced contracts and exploit the edge between your estimated true probability and the current market price.

DECISION FRAMEWORK:
1. Estimate the TRUE PROBABILITY of the event occurring based on all available analysis.
2. Compare your estimated probability against the current market price (from the odds report).
3. Calculate your EDGE: Edge = |Estimated Probability - Market Price|
4. Apply a MINIMUM EDGE THRESHOLD of {min_edge_threshold*100:.0f}%. If your edge is below {min_edge_threshold*100:.0f}%, you MUST recommend PASS regardless of direction.
5. For position sizing, use {kelly_fraction}x FRACTIONAL KELLY CRITERION:
   - For BUY_YES: Kelly fraction = (estimated_probability - market_price) / (1 - market_price)
   - For BUY_NO: Kelly fraction = (market_price - estimated_probability) / market_price
   - Position size = {kelly_fraction} * Kelly_fraction * bankroll (${bankroll:,.0f})
   - HARD CAP: Position size must NEVER exceed {max_position_pct*100:.0f}% of bankroll (${bankroll * max_position_pct:,.0f}) regardless of Kelly output.
   - This conservative sizing protects against estimation errors.

YOUR ANALYSIS MUST INCLUDE:
- Your estimated true probability (with reasoning)
- The current market price
- Your calculated edge (estimated probability minus market price)
- Whether the edge exceeds the {min_edge_threshold*100:.0f}% minimum threshold
- Position sizing reasoning using fractional Kelly (show the formula with numbers)
- Key risks that could invalidate your probability estimate

DECISION OPTIONS:
- BUY_YES: You believe the event is MORE likely than the market implies (your probability > market price + {min_edge_threshold*100:.0f}%)
- BUY_NO: You believe the event is LESS likely than the market implies (your probability < market price - {min_edge_threshold*100:.0f}%)
- PASS: Your edge is below {min_edge_threshold*100:.0f}%, or uncertainty is too high to have conviction

Do not forget to utilize lessons from past decisions to learn from your mistakes. Here are reflections from similar situations you traded in and the lessons learned:
{past_memory_str}

Always conclude your response with 'FINAL TRADE PROPOSAL: **BUY_YES/BUY_NO/PASS**' to confirm your recommendation.""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
