"""Terminal chat for manual testing: `uv run python -m agent.cli`. Every question makes a paid LLM call."""

from __future__ import annotations

import json

from agent.llm import LLMError, models_from_env
from agent.loop import Agent


def main() -> int:
    try:
        models = models_from_env()
    except LLMError as e:
        print(f"Cannot start: {e}")
        return 1
    agent = Agent(models)
    print(f"Airport investment agent ({' -> '.join(m.model_name for m in models)}). Empty line to quit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except EOFError:
            break
        if not question:
            break
        answer = agent.ask(question)
        for t in answer.trace:
            status = "error" if "error" in t.result else t.result.get("confidence")
            print(f"  [tool] {t.name}({json.dumps(t.arguments)}) -> {status}")
        print(f"\nagent> {answer.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
