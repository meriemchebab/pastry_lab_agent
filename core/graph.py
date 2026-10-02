"""LangGraph nodes, edges, and workflow compilation."""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from core.state import AgentBakingState

# Import all our node functions
from agents.router import router_node
from agents.search_agent import search_agent_node
from agents.math_agent import math_agent_node
from agents.vision_agent import vision_agent_node
from agents.copilot_agent import copilot_agent_node

# Import the tools each specialist is allowed to call
from tools.search_tool import web_search
from tools.baking_math import scale_pan_geometry, calculate_fat_substitution

workflow = StateGraph(AgentBakingState)
workflow.add_node("router", router_node)
workflow.add_node("search", search_agent_node)
workflow.add_node("math", math_agent_node)
workflow.add_node("vision", vision_agent_node)
workflow.add_node("copilot", copilot_agent_node)

# One ToolNode per specialist, so the loop can route back to the *right* agent.

workflow.add_node("search_tools", ToolNode([web_search]))
workflow.add_node("math_tools", ToolNode([scale_pan_geometry, calculate_fat_substitution]))

# Define the Flow
# start with router
workflow.add_edge(START, "router")


# A routing function to read the router's decision
def route_to_agent(state: AgentBakingState) -> str:
    return state.get("current_agent", "search")


# The router points conditionally to the specialists
workflow.add_conditional_edges(
    "router",
    route_to_agent,
    {
        "search": "search",
        "math": "math",
        "vision": "vision",
        "copilot": "copilot"
    }
)


workflow.add_conditional_edges(
    "search",
    tools_condition,
    {"tools": "search_tools", "__end__": END},
)
workflow.add_edge("search_tools", "search")

workflow.add_conditional_edges(
    "math",
    tools_condition,
    {"tools": "math_tools", "__end__": END},
)
workflow.add_edge("math_tools", "math")

# These two have no tools, so they finish immediately.
workflow.add_edge("vision", END)
workflow.add_edge("copilot", END)

# Compile the graph with memory preservation
memory = InMemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)
