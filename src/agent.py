"""
OpenRouter retention agent (OpenAI-compatible chat completions).

Uses only supplied scoring facts — no invented metrics. Falls back to the
rule-based explainer when the API key is missing or the call fails.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.explain import (
    explain_customer,
    explanation_facts_payload,
    format_explanation_text,
)
from src.scoring import ScoreResult

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """\
You are a concise e-commerce retention analyst for a student churn-prediction demo.

Rules:
- Use ONLY the facts JSON the user provides. Do not invent probabilities, F1 scores,
  feature values, or business KPIs that are not in the facts.
- Keep the reply short (roughly 120–180 words).
- Structure:
  1) One-sentence verdict (churn vs stay + risk band + P(churn) from facts).
  2) "Why" — 3–5 bullets grounded in this customer's feature values / driver comparisons.
  3) If predicted churn: "Retaining guide" — 3–5 bullets mapped to the supplied retention plays.
     If predicted stay: "Keep healthy" — 2–3 bullets; do not dump discounts.
- Prefer complaint recovery and service fixes before discounts.
- If a field is missing in the facts, skip it; do not guess.
"""


def _load_dotenv_if_available() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def get_api_key(*, explicit: str | None = None) -> str | None:
    """
    Resolve OpenRouter key from explicit arg, env, or Streamlit secrets.

    Order: ``explicit`` → ``OPENROUTER_API_KEY`` env → ``st.secrets``.
    """
    if explicit:
        return explicit.strip() or None

    _load_dotenv_if_available()
    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None and "OPENROUTER_API_KEY" in secrets:
            value = str(secrets["OPENROUTER_API_KEY"]).strip()
            return value or None
    except Exception:
        pass
    return None


def get_model_name(*, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    _load_dotenv_if_available()
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def build_user_prompt(facts: dict[str, Any]) -> str:
    from src.explain import dumps_facts

    return (
        "Write the retention note from these facts only.\n\n"
        f"```json\n{dumps_facts(facts)}\n```"
    )


def call_openrouter(
    facts: dict[str, Any],
    *,
    api_key: str,
    model: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Chat completion via OpenRouter; raises on HTTP / client errors."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for the retention agent. "
            "Install it (see requirements.txt) or use the rule-based fallback."
        ) from exc

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=timeout,
        default_headers={
            "HTTP-Referer": "https://github.com/ecommerce-customer-churn-prediction",
            "X-Title": "E-commerce Churn Retention Agent",
        },
    )
    response = client.chat.completions.create(
        model=get_model_name(explicit=model),
        temperature=0.2,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(facts)},
        ],
    )
    content = response.choices[0].message.content
    if not content or not str(content).strip():
        raise RuntimeError("OpenRouter returned an empty response")
    return str(content).strip()


def generate_retention_note(
    *,
    score: ScoreResult | dict[str, Any],
    raw_snapshot: dict[str, Any],
    preprocess_bundle: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Produce a retention note for the UI.

    Returns a dict with ``text``, ``source`` (``openrouter`` | ``rule_based``),
    ``explanation`` (structured rule-based payload), and optional ``error``.
    """
    explanation = explain_customer(
        score=score,
        raw_snapshot=raw_snapshot,
        preprocess_bundle=preprocess_bundle,
    )
    facts = explanation_facts_payload(
        score=score,
        raw_snapshot=raw_snapshot,
        preprocess_bundle=preprocess_bundle,
    )
    fallback_text = format_explanation_text(explanation)

    if not use_llm:
        return {
            "text": fallback_text,
            "source": "rule_based",
            "explanation": explanation,
            "facts": facts,
            "error": None,
        }

    key = get_api_key(explicit=api_key)
    if not key:
        return {
            "text": fallback_text,
            "source": "rule_based",
            "explanation": explanation,
            "facts": facts,
            "error": "OPENROUTER_API_KEY not set; showing rule-based guide.",
        }

    try:
        text = call_openrouter(facts, api_key=key, model=model)
        return {
            "text": text,
            "source": "openrouter",
            "explanation": explanation,
            "facts": facts,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — surface any API failure to the UI
        return {
            "text": fallback_text,
            "source": "rule_based",
            "explanation": explanation,
            "facts": facts,
            "error": f"OpenRouter call failed ({exc}); showing rule-based guide.",
        }
