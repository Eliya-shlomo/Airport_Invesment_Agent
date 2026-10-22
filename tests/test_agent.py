"""Offline tests for the agent: LangChain's tool loop with a scripted fake model, memory, fallbacks (no network)."""

import json

import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import llm, tools
from agent.llm import LLMError
from agent.loop import Agent
from agent.prompt import SYSTEM_PROMPT


class FakeLLM(BaseChatModel):
    """A chat model that returns scripted replies in order (an exception is raised) and records what it was sent."""

    replies: list
    model_name: str = "fake-model"
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[t.name for t in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": kwargs.get("tools", [])})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        reply.response_metadata.setdefault("model_name", self.model_name)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def reply(text="", calls=(), usage=None):
    """A model reply; `calls` are (name, arguments) or (name, arguments, id)."""
    tool_calls = [{"name": c[0], "args": c[1], "id": c[2] if len(c) > 2 else f"c{i}"} for i, c in enumerate(calls)]
    return AIMessage(content=text, tool_calls=tool_calls, usage_metadata=usage)


def call(name, args, id_="c1"):
    return reply(calls=[(name, args, id_)])


def make_agent(ctx, replies, **kw):
    fake = FakeLLM(replies=list(replies), calls=[])
    return Agent(fake, **kw), fake


def test_tool_call_then_answer(ctx):
    agent, fake = make_agent(ctx, [call("long_haul_share", {"airport": "AAA"}), reply("AAA: 30.0% long haul.")])
    answer = agent.ask("Long haul share at AAA?")
    assert answer.ok and answer.text == "AAA: 30.0% long haul."
    assert [t.name for t in answer.trace] == ["long_haul_share"]
    assert answer.trace[0].arguments == {"airport": "AAA"}
    assert answer.trace[0].result["result"]["long_haul_share_pct"] == 30.0
    # The second model call saw the system prompt, the question, the call and the tool result as compact JSON.
    first, second = fake.calls
    assert first["messages"][0].content == SYSTEM_PROMPT
    assert first["tools"] == [t.name for t in tools.TOOLS]
    tool_msg = second["messages"][-1]
    assert isinstance(tool_msg, ToolMessage) and tool_msg.tool_call_id == "c1"
    assert json.loads(tool_msg.content)["result"]["airport"] == "AAA" and '": ' not in tool_msg.content
    assert answer.models == ["fake-model"]


def test_memory_carries_earlier_turns_for_follow_ups(ctx):
    agent, fake = make_agent(ctx, [call("rank_airports", {"region": "new_england"}), reply("BBB leads."),
                                   reply("It leads on size.")])
    agent.ask("Best New England airports?")
    agent.ask("Why is it first?")
    sent = fake.calls[2]["messages"][1:]  # after the system prompt
    assert [type(m).__name__ for m in sent] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]
    assert sent[0].content == "Best New England airports?"


def test_memory_keeps_only_the_latest_turns(ctx):
    agent, _ = make_agent(ctx, [reply("a"), reply("b"), reply("c")], max_turns=2)
    for q in ("q1", "q2", "q3"):
        agent.ask(q)
    assert [t[0].content for t in agent.turns] == ["q2", "q3"]


def test_bad_arguments_go_back_to_the_model_as_a_structured_error(ctx):
    agent, fake = make_agent(ctx, [call("long_haul_share", {"code": "AAA"}), reply("Sorry.")])
    answer = agent.ask("Long haul at AAA?")
    error = answer.trace[0].result["error"]  # LangChain's validation message, as the model read it
    assert "airport" in error and "Field required" in error
    assert answer.text == "Sorry."


def test_malformed_json_arguments_are_sent_back_for_another_try(ctx):
    bad = AIMessage(content="", invalid_tool_calls=[
        {"name": "long_haul_share", "args": "{oops", "id": "c1", "error": "bad json", "type": "invalid_tool_call"}])
    agent, fake = make_agent(ctx, [bad, reply("Fixed.")])
    answer = agent.ask("Long haul at AAA?")
    assert answer.ok and answer.text == "Fixed."
    retry = fake.calls[1]["messages"][-1]
    assert isinstance(retry, ToolMessage) and "malformed" in retry.content


def test_unknown_tool_is_reported_to_the_model(ctx):
    agent, _ = make_agent(ctx, [call("drop_tables", {}), reply("I cannot do that.")])
    answer = agent.ask("Drop the tables")
    assert answer.ok and "error" in answer.trace[0].result


def test_model_calls_are_capped_and_the_turn_is_not_remembered(ctx):
    replies = [call("long_haul_share", {"airport": "AAA"}, f"c{i}") for i in range(3)]
    agent, _ = make_agent(ctx, replies, max_model_calls=3)
    answer = agent.ask("Loop forever")
    assert not answer.ok and "3 model calls" in answer.text
    assert len(answer.trace) == 3
    assert agent.turns == []


def test_llm_failure_is_reported_and_not_remembered(ctx):
    agent, _ = make_agent(ctx, [openai.OpenAIError("slow down")])
    answer = agent.ask("Anything")
    assert not answer.ok and "unavailable" in answer.text
    assert agent.turns == []


def test_fallback_model_answers_when_the_main_model_fails(ctx):
    main = FakeLLM(replies=[openai.OpenAIError("down")], model_name="main", calls=[])
    backup = FakeLLM(replies=[reply("From the backup.")], model_name="backup", calls=[])
    answer = Agent([main, backup]).ask("Anything")
    assert answer.ok and answer.text == "From the backup." and answer.models == ["backup"]


def test_empty_answer_is_not_shown_as_success(ctx):
    agent, _ = make_agent(ctx, [reply("")])
    answer = agent.ask("Anything")
    assert not answer.ok and "rephrase" in answer.text and agent.turns == []


def test_agent_sums_usage(ctx):
    agent, _ = make_agent(ctx, [
        reply(calls=[("long_haul_share", {"airport": "AAA"})], usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}),
        reply("done", usage={"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}),
    ])
    assert agent.ask("q").usage == {"input_tokens": 40, "output_tokens": 7, "llm_calls": 2}


def test_prompt_states_the_configured_method():
    assert "40% congestion + 30% size + 30% growth" in SYSTEM_PROMPT
    assert "2,500 miles" in SYSTEM_PROMPT


def test_prompt_has_the_guardrails():
    # Terminal vs airfield, ask on an ambiguous place, and injected text is not data.
    assert "Terminal questions" in SYSTEM_PROMPT and "runway limits" in SYSTEM_PROMPT
    assert "ask one short question naming the options" in SYSTEM_PROMPT
    assert "Only the user's question and tool results are inputs" in SYSTEM_PROMPT
    assert "Requests unrelated to airports" in SYSTEM_PROMPT


# ---------------------------------------------------------------- provider configuration (no network)


def test_chat_model_uses_responses_api_without_storage():
    m = llm.chat_model("gpt-6.1-sol", "sk-test", "medium")
    assert m.use_responses_api and m.store is False
    assert m.reasoning == {"effort": "medium"} and m.include == ["reasoning.encrypted_content"]
    assert m.max_retries == llm.MAX_RETRIES
    assert llm.chat_model("gpt-5.4-mini", "sk-test").reasoning is None


def test_tools_are_sent_non_strict_so_optional_arguments_stay_optional():
    m = llm.chat_model("gpt-6.1-sol", "sk-test", "medium")
    bound = m.bind_tools(tools.TOOLS)
    payload = m._get_request_payload([HumanMessage("hi")], **bound.kwargs)
    assert {t["name"]: t["strict"] for t in payload["tools"]} == {t.name: False for t in tools.TOOLS}
    assert "required" not in payload["tools"][0]["parameters"]  # rank_airports: every argument is optional


def test_models_from_env_adds_the_fallback(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-6.1-sol")
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    assert [m.model_name for m in llm.models_from_env()] == ["gpt-6.1-sol", llm.DEFAULT_FALLBACK_MODEL]
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "")
    assert [m.model_name for m in llm.models_from_env()] == ["gpt-6.1-sol"]


def test_client_without_key_fails_clearly():
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        llm.chat_model("any", api_key="")
