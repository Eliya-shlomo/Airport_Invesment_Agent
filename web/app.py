"""Chat web app: `uv run uvicorn web.app:app`. One conversation per browser (session cookie), kept in memory.

POST /api/chat answers with one JSON object: the answer text, the tool trace (for the show-work panel), token usage,
the models that answered and the elapsed time.
"""

from __future__ import annotations

import functools
import logging
import secrets
import time
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agent.llm import LLMError, models_from_env
from agent.loop import Agent

log = logging.getLogger("web")

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "sid"
MAX_QUESTION_CHARS = 1000


class ChatRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is empty")
        return v


@functools.cache
def _models() -> list:
    """Created on the first question, so the server starts without a key and says so per request."""
    return models_from_env()  # raises LLMError when the key is missing (not cached, so a fixed .env works)


def default_agent() -> Agent:
    return Agent(_models())


def create_app(new_agent: Callable[[], Agent] = default_agent) -> FastAPI:
    app = FastAPI(title="Airport Investment Intelligence Agent", docs_url=None, redoc_url=None)
    conversations: dict[str, Agent] = {}  # session id -> conversation; lost on restart

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request) -> JSONResponse:
        sid = request.cookies.get(SESSION_COOKIE)
        if sid not in conversations:
            try:
                agent = new_agent()
            except LLMError as e:
                return JSONResponse({"error": f"The agent is not configured: {e}"}, status_code=503)
            sid = secrets.token_urlsafe(24)
            conversations[sid] = agent
        started = time.monotonic()
        try:
            answer = conversations[sid].ask(body.question)
            response = JSONResponse({
                "text": answer.text, "ok": answer.ok, "usage": answer.usage, "models": answer.models,
                "elapsed_s": round(time.monotonic() - started, 1),
                "trace": [{"name": t.name, "arguments": t.arguments, "result": t.result} for t in answer.trace],
            })
        except Exception:  # a bug must not reach the page; details go to the server log only
            log.exception("agent failed")
            response = JSONResponse({"error": "Something went wrong while answering. Please try again."},
                                    status_code=500)
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
        return response

    @app.post("/api/reset")
    def reset(request: Request) -> JSONResponse:
        """Start a new conversation: forget the old one and drop the session cookie."""
        conversations.pop(request.cookies.get(SESSION_COOKIE, ""), None)
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE)
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
