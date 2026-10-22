"""The agent: LangChain's agent graph (model -> tools -> model ... -> answer) plus conversation memory.

LangChain runs the tool loop; this module adds what the project needs around it: a cap on model calls, a fallback
model, the tool trace for the show-work panel and the evals, and memory of whole turns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelFallbackMiddleware, after_model
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.llm import PROVIDER_ERRORS
from agent.prompt import SYSTEM_PROMPT
from agent.tools import TOOLS

MAX_MODEL_CALLS = 6  # per question; the last one may still call tools, then the question stops
MAX_REMEMBERED_TURNS = 12  # whole question-answer turns kept as conversation memory


@after_model(can_jump_to=["model"])
def retry_malformed_tool_calls(state: dict, runtime) -> dict | None:
    """When the model's only tool calls have arguments that are not valid JSON, ask it again instead of ending the
    turn. LangChain tells the model which calls were malformed; the model-call cap still bounds the retries."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.invalid_tool_calls and not last.tool_calls:
        return {"jump_to": "model"}
    return None


@dataclass
class ToolTrace:
    name: str
    arguments: dict | None
    result: dict


@dataclass
class AgentAnswer:
    text: str
    trace: list[ToolTrace] = field(default_factory=list)
    ok: bool = True
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
    models: list[str] = field(default_factory=list)  # which models answered; more than one means the fallback ran


class Agent:
    """One conversation: memory of earlier turns plus LangChain's tool loop."""

    def __init__(self, models: BaseChatModel | list[BaseChatModel], max_model_calls: int = MAX_MODEL_CALLS,
                 max_turns: int = MAX_REMEMBERED_TURNS):
        models = models if isinstance(models, list) else [models]
        middleware = [ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="error"),
                      retry_malformed_tool_calls]
        if len(models) > 1:
            middleware.append(ModelFallbackMiddleware(*models[1:]))
        self.graph = create_agent(models[0], TOOLS, system_prompt=SYSTEM_PROMPT, middleware=middleware)
        self.max_model_calls = max_model_calls
        self.max_turns = max_turns
        self.turns: list[list[BaseMessage]] = []  # each turn: question, tool calls and results, final answer

    def history(self) -> list[BaseMessage]:
        return [m for turn in self.turns for m in turn]

    def ask(self, question: str) -> AgentAnswer:
        """Answer one question.

        A turn that fails (provider down, too many model calls, empty answer) is not remembered, so the next question
        starts from the last good state and no unanswered tool call is left in memory. The graph is streamed rather
        than invoked so that the tool calls made before a failure are still at hand for the trace.
        """
        sent = self.history() + [HumanMessage(question)]
        new: list[BaseMessage] = []
        try:
            for state in self.graph.stream({"messages": sent}, stream_mode="values"):
                new = state["messages"][len(sent):]
        except ModelCallLimitExceededError:
            return _answer(f"I could not finish this answer within {self.max_model_calls} model calls. Please "
                           "narrow the question (for example one airport or one region).", new, ok=False)
        except PROVIDER_ERRORS as e:
            return _answer(f"The language model is unavailable right now ({type(e).__name__}). Please try again.",
                           new, ok=False)
        text = new[-1].text if new and isinstance(new[-1], AIMessage) else ""
        if not text.strip():
            return _answer("I could not produce an answer to that. Please rephrase the question.", new, ok=False)
        self.turns.append(sent[-1:] + new)
        del self.turns[:-self.max_turns]
        return _answer(text, new)


def _answer(text: str, messages: list[BaseMessage], ok: bool = True) -> AgentAnswer:
    """Tool trace, token usage and model names from the messages of one turn."""
    replies = [m for m in messages if isinstance(m, AIMessage)]
    args = {c["id"]: c["args"] for m in replies for c in m.tool_calls}
    trace = [ToolTrace(m.name, args.get(m.tool_call_id), _tool_result(m))
             for m in messages if isinstance(m, ToolMessage)]
    usage = {"input_tokens": sum((m.usage_metadata or {}).get("input_tokens", 0) for m in replies),
             "output_tokens": sum((m.usage_metadata or {}).get("output_tokens", 0) for m in replies),
             "llm_calls": len(replies)}
    models = list(dict.fromkeys(m.response_metadata["model_name"] for m in replies
                                if m.response_metadata.get("model_name")))
    return AgentAnswer(text, trace, ok, usage, models)


def _tool_result(message: ToolMessage) -> dict:
    """The tool's JSON result as the model read it. LangChain's own errors (invalid arguments, unknown tool) are
    plain text."""
    try:
        return json.loads(message.text)
    except ValueError:
        return {"error": message.text, "valid_options": []}
