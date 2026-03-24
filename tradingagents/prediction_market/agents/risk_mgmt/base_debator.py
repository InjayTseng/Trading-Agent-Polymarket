"""Shared factory for risk management debator agents.

All three debators (aggressive, conservative, neutral) share identical structure —
only the system prompt, role name, and which "other" responses they read differ.
"""

from typing import Dict, List


def create_risk_debator(
    llm,
    role: str,
    prompt_template: str,
    own_history_key: str,
    own_response_key: str,
    other_response_keys: List[str],
):
    """Create a risk debator node with the given role and prompt.

    Args:
        llm: Language model instance
        role: Display name (e.g., "Aggressive Analyst")
        prompt_template: Prompt with {trader_decision}, {event_report}, {odds_report},
                         {information_report}, {sentiment_report}, {history},
                         {other_responses} placeholders
        own_history_key: State key for this debator's history (e.g., "aggressive_history")
        own_response_key: State key for this debator's latest response (e.g., "current_aggressive_response")
        other_response_keys: State keys for the other debators' latest responses
    """
    speaker_label = role.split()[0]  # "Aggressive", "Conservative", "Neutral"

    def debator_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        own_history = risk_debate_state.get(own_history_key, "")

        other_responses = " ".join(
            risk_debate_state.get(k, "") for k in other_response_keys
        )

        prompt = prompt_template.format(
            trader_decision=state["trader_investment_plan"],
            event_report=state["event_report"],
            odds_report=state["odds_report"],
            information_report=state["information_report"],
            sentiment_report=state["sentiment_report"],
            history=history,
            other_responses=other_responses,
        )

        response = llm.invoke(prompt)
        argument = f"{role}: {response.content}"

        new_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": speaker_label,
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }
        # Override own fields
        new_state[own_history_key] = own_history + "\n" + argument
        new_state[own_response_key] = argument

        return {"risk_debate_state": new_state}

    return debator_node
