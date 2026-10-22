# Airport Investment Intelligence Agent

A chat agent that helps analysts at an airport-investment firm find US airports where an expansion is most likely to pay off because demand is outgrowing capacity. It pulls public aviation data into a local database, scores and ranks airports with deterministic code, explains its reasoning, states its assumptions and uncertainty, and handles follow-up questions. Voice dictation is a bonus.

**The design document** (scoring methodology, tradeoffs, where AI is used): [docs/design.md](docs/design.md)

## What it answers

- Which airports in New England are strong candidates for terminal expansion?
- Compare LA and Santa Ana airport congestion levels.
- What is the percentage of long haul flights out of Anchorage airport?
- What is the unmet flight demand in SFO airport and why?
- Follow-ups: "Why is the second one ranked below Boston?", "What about Hartford?"

Out of scope, and said plainly by the agent: non-US airports, construction cost, ROI in dollars or financing, forecasts, investment advice, and real-time operations.

## How it works

1. The language model reads the question, picks one of six tools and fills its arguments.
2. The tool (plain Python) computes every number from a local DuckDB database and returns it with its method, assumptions, sources, period, confidence and warnings.
3. The model explains the returned numbers. **It never computes one.**

The expansion opportunity score is 40% congestion + 30% size + 30% growth, each a national percentile among 140 FAA hub airports. Unmet demand is reported as evidence (a count of four signals), never as a passenger number. Congestion is measured on the airfield, so for terminal questions the agent says that a runway-limited airport may need airfield rather than terminal investment. Every answer cites its datasets, periods and fetch dates, and the chat page shows each tool call under "Show work".

Where to look first: [agent/prompt.py](agent/prompt.py) (system prompt), [agent/tools/](agent/tools/__init__.py) (the six tools, one file each; [common.py](agent/tools/common.py) holds the result contract), [analytics/scoring.py](analytics/scoring.py) (the score) and [analytics/config.py](analytics/config.py) (every weight and threshold).

## Run it

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env                 # then put your key in .env (OPENAI_API_KEY)
uv run uvicorn web.app:app --reload  # chat UI at http://localhost:8000
```

The built database (`data/processed/airports.duckdb`, about 3 MB, built on 2026-10-22) is in the repository, so the agent runs without downloading anything. To rebuild it from the public sources:

```bash
uv run python -m sources.download    # about 350 MB of public data, no keys; skips files already there
uv run python -m sources.build       # rebuilds data/processed/airports.duckdb in about 10 seconds
```

Other commands:

```bash
uv run python -m agent.cli           # terminal chat (paid model calls)
uv run pytest                        # offline unit tests: no network, no key
uv run python -m evals.run           # golden-question evals: prints the plan; --confirm makes the paid calls
```

Questions to the agent make paid calls to the OpenAI API. The default model is `gpt-6.1-sol` with medium reasoning, and the fallback is `gpt-5.4-mini`. `LLM_MODEL`, `LLM_REASONING_EFFORT` and `LLM_FALLBACK_MODEL` change them; an empty fallback disables it.

## Layout

```
web/         FastAPI app and one static chat page (voice dictation through the browser)
agent/       LangChain agent and memory, tools, system prompt, model setup
analytics/   pure, deterministic scoring, confidence and airfare signals; constants in config.py
sources/     download and build scripts (per-airport summaries in SQL); the only layer that touches the network
evals/       golden questions and answer checks for the full agent
tests/       offline unit tests with small hand-built fixtures
docs/        design document
```

## Data and results

Public sources, no keys: BTS On-Time Reporting (12 months to 2026-08), FAA enplanements (CY2025 and CY2024), the DOT airfare table 1a (to 2026 Q1) and OurAirports.

On 25 golden questions (the brief, follow-ups, single-metric lists, out-of-scope requests, prompt injection, an ambiguous place name, missing data, an unknown code), the latest run of `gpt-6.1-sol` passed 24, picked the right tool 25 times, and used no number that is not in a tool result (mean 11.0 s, $0.014 per question). Across the three full runs it passed 23 to 24, with the right tool and no ungrounded number every time. The checks are regular expressions: three failures were correct answers worded in a way a pattern missed (the patterns are fixed and those answers kept as test cases), and the one real failure (a poem written for an off-topic request) led to a prompt rule; that question has passed in every run since. Raw results of every run: `evals/results/`.

## Known limits and next steps

- No seat or gate data, so congestion is measured by its effects (on the airfield, not in the terminal) and unmet demand is an indirect estimate.
- On-Time covers domestic flights of reporting carriers only; growth covers one year; the weights are a judgement (every ranking returns a rank range under weight changes).
- All sources were downloaded with `sources.download` and the database built on 2026-10-22. Voice was not tested on a real microphone, Safari or Firefox.
- Next, in order: a longer growth history, seat data from BTS T-100, validation against airports that actually expanded, a scope classifier in front of the model, run-to-run variation of the evals, and persistent conversations.
