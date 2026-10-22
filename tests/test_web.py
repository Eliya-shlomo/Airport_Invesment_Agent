"""Offline tests for the chat web app: fake LLM, in-memory tool context, FastAPI TestClient (no network)."""

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from agent.llm import LLMError
from agent.loop import Agent
from tests.test_agent import FakeLLM, call, reply
from web.app import MAX_QUESTION_CHARS, SESSION_COOKIE, create_app


def make_client(ctx, scripts):
    """`scripts` is a list of reply lists; each new session gets the next script. Returns (client, fakes)."""
    fakes = []
    pending = list(scripts)

    def factory():
        fake = FakeLLM(replies=pending.pop(0), calls=[])
        fakes.append(fake)
        return Agent(fake)

    return TestClient(create_app(factory)), fakes


LONG_HAUL = [call("long_haul_share", {"airport": "AAA"}), reply("AAA: 30.0% long haul.")]


def questions_sent(fake, n):
    """The user questions in the n-th model call of a fake."""
    return [m.content for m in fake.calls[n]["messages"] if isinstance(m, HumanMessage)]


def test_index_page_is_served(ctx):
    client, _ = make_client(ctx, [])
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/static/app.js").status_code == 200


def test_chat_returns_the_answer_with_trace(ctx):
    client, _ = make_client(ctx, [LONG_HAUL])
    r = client.post("/api/chat", json={"question": "Long haul share at AAA?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    answer = r.json()
    assert answer["ok"] and answer["text"] == "AAA: 30.0% long haul."
    trace = answer["trace"][0]
    assert trace["name"] == "long_haul_share" and trace["arguments"] == {"airport": "AAA"}
    # The trace carries the tool contract the page shows under "show work".
    for key in ("confidence", "warnings", "sources", "assumptions", "method", "coverage"):
        assert key in trace["result"]
    assert trace["result"]["result"]["long_haul_share_pct"] == 30.0
    assert answer["elapsed_s"] >= 0 and answer["usage"]["llm_calls"] == 2


def test_same_cookie_keeps_memory_and_a_new_client_gets_a_new_conversation(ctx):
    client, fakes = make_client(ctx, [[reply("one"), reply("two")], [reply("fresh")]])
    first = client.post("/api/chat", json={"question": "q1"})
    assert SESSION_COOKIE in first.cookies and "httponly" in first.headers["set-cookie"].lower()
    client.post("/api/chat", json={"question": "q2"})  # TestClient sends the cookie back
    assert questions_sent(fakes[0], 1) == ["q1", "q2"]

    other = TestClient(client.app)  # no cookie: a different browser
    other.post("/api/chat", json={"question": "hello"})
    assert len(fakes) == 2 and questions_sent(fakes[1], 0) == ["hello"]


def test_reset_starts_a_new_conversation(ctx):
    client, fakes = make_client(ctx, [[reply("one")], [reply("two")]])
    client.post("/api/chat", json={"question": "q1"})
    assert client.post("/api/reset").json() == {"ok": True}
    client.post("/api/chat", json={"question": "q2"})
    assert len(fakes) == 2 and questions_sent(fakes[1], 0) == ["q2"]


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "   "}, {}, {"question": "x" * (MAX_QUESTION_CHARS + 1)}])
def test_invalid_questions_are_rejected(ctx, body):
    client, fakes = make_client(ctx, [[reply("never")]])
    assert client.post("/api/chat", json=body).status_code == 422
    assert fakes == []  # no session, no model call


def test_provider_failure_is_reported_in_the_answer_and_not_remembered(ctx):
    client, fakes = make_client(ctx, [[openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")), reply("ok now")]])
    bad = client.post("/api/chat", json={"question": "q1"}).json()
    assert not bad["ok"] and "APIConnectionError" in bad["text"]
    client.post("/api/chat", json={"question": "q1 again"})
    assert questions_sent(fakes[0], 1) == ["q1 again"]


def test_unexpected_exception_becomes_an_error_without_details(ctx):
    boom = FakeLLM(replies=[RuntimeError("secret internal detail")], calls=[])
    client = TestClient(create_app(lambda: Agent(boom)))
    r = client.post("/api/chat", json={"question": "q"})
    assert r.status_code == 500
    assert r.json() == {"error": "Something went wrong while answering. Please try again."}


def test_missing_key_gives_503_with_a_clear_message(ctx):
    def factory():
        raise LLMError("OPENAI_API_KEY is not set")

    r = TestClient(create_app(factory)).post("/api/chat", json={"question": "q"})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["error"]
