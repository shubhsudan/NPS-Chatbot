"""
topic_guard.py — Decide whether a query is within scope for the NPS chatbot.

Scope is intentionally broad: any question a person could reasonably ask
when trying to understand, plan, or manage their NPS account or retirement
savings in India. This includes general financial/retirement questions that
an NPS subscriber might have, even if they don't mention "NPS" explicitly.

Examples IN scope:
  - "What happens to my money when I retire?"
  - "How can I save tax on my salary?"
  - "Can I withdraw before 60?"
  - "What is a good pension plan in India?"

Examples OUT of scope:
  - "What is the capital of France?"
  - "Write me a poem"
  - "Who won the cricket match?"

Always uses the LLM classifier — no keyword gating — so naturally phrased
questions are never incorrectly blocked.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_CLASSIFIER_SYSTEM = (
    "You are a query classifier for an NPS (National Pension System) chatbot. "
    "Reply YES if the query is something a user of an Indian pension/retirement "
    "savings chatbot might reasonably ask — including questions about NPS, PFRDA, "
    "PRAN, retirement planning, pension withdrawals, annuity, tax saving on salary, "
    "or general financial questions relevant to an Indian salaried or self-employed person. "
    "Reply NO only if the query is clearly unrelated to finance, retirement, or pension "
    "(e.g. sports, geography, entertainment, coding). "
    "When in doubt, reply YES. Reply with exactly one word: YES or NO."
)

_groq_client: Groq | None = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def is_in_scope(query: str) -> bool:
    """
    Returns True if the query is within scope for the NPS chatbot.
    Uses LLM classification with a generous scope definition.
    Defaults to True on any API failure so users are never silently blocked.
    """
    try:
        resp = _get_groq().chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return True  # fail open — never block users due to API errors
