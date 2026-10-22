"""Shared fixtures: the tools run on the tiny fixture tables, never on the real database."""

import pytest

from agent import tools
from tests.fixtures.analytics_tables import tiny_tables


@pytest.fixture(scope="session")
def tiny_ctx():
    return tools.ToolContext.from_tables(tiny_tables())


@pytest.fixture
def ctx(tiny_ctx):
    tools.set_context(tiny_ctx)
    yield tiny_ctx
    tools.set_context(None)
