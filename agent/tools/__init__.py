"""The agent's six tools, one file each. Every tool is a LangChain tool made by `airport_tool` (common.py): its
docstring is what the model reads, its typed signature is the argument schema, and enums come from
analytics/config.py so the model can only ask for what the code supports.

Reading order: common.py (data, decorator, contract, lookup), then scores.py, congestion.py, long_haul.py and
unmet_demand.py.
"""

from agent.tools.common import ToolContext, ToolError, get_context, set_context
from agent.tools.congestion import compare_congestion
from agent.tools.long_haul import long_haul_share
from agent.tools.scores import airport_scores, rank_airports, top_airports_by
from agent.tools.unmet_demand import unmet_demand

TOOLS = [rank_airports, top_airports_by, airport_scores, compare_congestion, long_haul_share, unmet_demand]

__all__ = ["TOOLS", "ToolContext", "ToolError", "get_context", "set_context"]
