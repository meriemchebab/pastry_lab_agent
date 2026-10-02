"""LangGraph nodes, edges, and workflow compilation."""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from core.state import AgentBakingState

# Import all our node functions
from agents.router import router_node
from agents.search_agent import search_agent_node
from agents.math_agent import math_agent_node
from agents.vision_agent import vision_agent_node
from agents.copilot_agent import copilot_agent_node

workflow = StateGraph(AgentBakingState)
workflow.add_node("router", router_node)
workflow.add_node("search", search_agent_node)
workflow.add_node("math", math_agent_node)
workflow.add_node("vision", vision_agent_node)
workflow.add_node("copilot", copilot_agent_node)

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

# Terminate the turn after a specialist finishes
workflow.add_edge("search", END)
workflow.add_edge("math", END)
workflow.add_edge("vision", END)
workflow.add_edge("copilot", END)

# Compile the graph with memory preservation
memory = InMemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)