"""The only module that imports the LLM provider package (langchain-openai). Switching provider changes this file only.

Models run through the OpenAI Responses API with nothing stored server-side (store=False). Reasoning items come back
encrypted and LangChain sends them back with the tool outputs. Fallbacks, from the inside out:
the client retries transient failures (connection, 429, 5xx) MAX_RETRIES times; then a second model
(LLM_FALLBACK_MODEL) takes over; when that fails too, `agent/loop.py` reports the error and forgets the turn.
"""

from __future__ import annotations

import os

import openai
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_FALLBACK_MODEL = "gpt-5.4-mini"
REQUEST_TIMEOUT_S = 90
MAX_RETRIES = 2

PROVIDER_ERRORS: tuple[type[Exception], ...] = (openai.OpenAIError,)


class LLMError(Exception):
    """The agent cannot be configured (missing key)."""


class _ChatOpenAI(ChatOpenAI):
    """ChatOpenAI with non-strict function tools. The Responses API treats a function without a `strict` field as
    strict, and strict mode makes every argument required: the model then sends `region` and `states: []` together,
    which `rank_airports` rejects (seen in the evals). Non-strict keeps optional arguments optional."""

    def bind_tools(self, tools, **kwargs):
        kwargs.setdefault("strict", False)
        return super().bind_tools(tools, **kwargs)


def chat_model(model: str, api_key: str, reasoning_effort: str | None = None) -> BaseChatModel:
    """One OpenAI model through the Responses API. Never logs the key."""
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set (put it in .env or the environment)")
    kwargs: dict = {"model": model, "api_key": api_key, "use_responses_api": True, "store": False,
                    "timeout": REQUEST_TIMEOUT_S, "max_retries": MAX_RETRIES}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
        kwargs["include"] = ["reasoning.encrypted_content"]  # with store=False, reasoning travels encrypted
    return _ChatOpenAI(**kwargs)


def models_from_env() -> list[BaseChatModel]:
    """[main model, fallback model] from OPENAI_API_KEY, LLM_MODEL, LLM_REASONING_EFFORT and LLM_FALLBACK_MODEL."""
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "")
    models = [chat_model(os.environ.get("LLM_MODEL") or DEFAULT_MODEL, key,
                         os.environ.get("LLM_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT)]
    fallback = os.environ.get("LLM_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
    if fallback and fallback != models[0].model_name:
        models.append(chat_model(fallback, key))  # no reasoning effort: it must work on any model
    return models
