"""Offline tests for LangGraph wiring and turn-to-turn state handling."""

import os
from dataclasses import replace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition


class _OfflineModel:
    """Stand-in for a Gemini model; graph tests replace every invoked node."""

    def bind_tools(self, tools):
        return self


# Agent modules currently construct Gemini clients at import time. Supply a
# temporary key and replace the client constructor so collection stays offline.
with patch.dict(os.environ, {"GOOGLE_API_KEY": "offline-test-key"}):
    with patch.object(ChatGoogleGenerativeAI, "__init__", lambda self, **kwargs: None):
        import core.graph as graph_module

from core.graph import route_to_agent


def test_route_to_agent_uses_current_agent():
    assert route_to_agent({"current_agent": "math"}) == "math"


def test_route_to_agent_defaults_to_search():
    assert route_to_agent({}) == "search"


def test_graph_routes_to_selected_node_and_returns_its_message(monkeypatch):
    def fake_router(state):
        return {"current_agent": "math"}

    def fake_math(state):
        assert isinstance(state["messages"][-1], HumanMessage)
        return {"messages": [AIMessage(content="The pans are close in area.")]}

    monkeypatch.setattr(graph_module, "router_node", fake_router)
    monkeypatch.setattr(graph_module, "math_agent_node", fake_math)
    graph_module.workflow.nodes["router"] = replace(
        graph_module.workflow.nodes["router"], runnable=fake_router
    )
    graph_module.workflow.nodes["math"] = replace(
        graph_module.workflow.nodes["math"], runnable=fake_math
    )
    test_graph = graph_module.workflow.compile()

    result = test_graph.invoke({"messages": [HumanMessage(content="Scale this cake")]})

    assert result["current_agent"] == "math"
    assert result["messages"][-1].content == "The pans are close in area."


def test_compiled_graph_keeps_state_between_turns(monkeypatch):
    def fake_router(state):
        return {"current_agent": "copilot"}

    def fake_copilot(state):
        return {"messages": [AIMessage(content=f"Turn {len(state['messages'])}")]}

    monkeypatch.setattr(graph_module, "router_node", fake_router)
    monkeypatch.setattr(graph_module, "copilot_agent_node", fake_copilot)
    graph_module.workflow.nodes["router"] = replace(
        graph_module.workflow.nodes["router"], runnable=fake_router
    )
    graph_module.workflow.nodes["copilot"] = replace(
        graph_module.workflow.nodes["copilot"], runnable=fake_copilot
    )
    test_graph = graph_module.workflow.compile(checkpointer=graph_module.InMemorySaver())
    config = {"configurable": {"thread_id": "graph-test-thread"}}

    first = test_graph.invoke(
        {"messages": [HumanMessage(content="Start baking")]}, config=config
    )
    second = test_graph.invoke(
        {"messages": [HumanMessage(content="Done mixing")], "current_step": 2},
        config=config,
    )

    assert len(first["messages"]) == 2
    assert len(second["messages"]) == 4
    assert [message.content for message in second["messages"]] == [
        "Start baking",
        "Turn 1",
        "Done mixing",
        "Turn 3",
    ]
    assert second["current_step"] == 2


def test_tool_edges_can_terminate_without_calling_a_tool(monkeypatch):
    """Regression: a tool agent that answers immediately must reach END.

    tools_condition returns the literal "__end__" rather than the END sentinel.
    If the conditional-edge path map omits that key, the graph raises
    KeyError('__end__') on the first reply that needs no tool call.
    """
    def fake_router(state):
        return {"current_agent": "search"}

    def fake_search(state):
        return {"messages": [AIMessage(content="### Quick Cake\n")]}

    monkeypatch.setattr(graph_module, "router_node", fake_router)
    monkeypatch.setattr(graph_module, "search_agent_node", fake_search)
    graph_module.workflow.nodes["router"] = replace(
        graph_module.workflow.nodes["router"], runnable=fake_router
    )
    graph_module.workflow.nodes["search"] = replace(
        graph_module.workflow.nodes["search"], runnable=fake_search
    )
    test_graph = graph_module.workflow.compile()

    result = test_graph.invoke({"messages": [HumanMessage(content="cocoa?")]})

    assert result["messages"][-1].content == "### Quick Cake\n"


def test_search_agent_loops_through_tools_before_ending(monkeypatch):
    """The model asks for a tool, gets a result, then writes the final answer."""
    from langchain_core.tools import tool as make_tool
    from langgraph.prebuilt import ToolNode

    passes = {"n": 0}

    @make_tool
    def fake_tool(query: str) -> str:
        """Search for a recipe.

        Args:
            query: what to search for.
        """
        return f"results for {query}"

    tool_node = ToolNode([fake_tool])

    def fake_search(state):
        passes["n"] += 1
        if passes["n"] == 1:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "fake_tool",
                                "args": {"query": "cocoa cake"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        return {
            "messages": [AIMessage(content="Here is your cocoa cake.")],
            "tool_trace": ["fake_tool"],
        }

    stub = StateGraph(graph_module.AgentBakingState)
    stub.add_node("search", fake_search)
    stub.add_node("tools", tool_node)
    stub.add_edge(START, "search")
    stub.add_conditional_edges("search", tools_condition, {"tools": "tools", "__end__": END})
    stub.add_edge("tools", "search")
    compiled = stub.compile()

    result = compiled.invoke({"messages": [HumanMessage(content="cocoa?")]})

    assert passes["n"] == 2, "agent should run once to call the tool, once to answer"
    assert result["messages"][-1].content == "Here is your cocoa cake."
    assert result["tool_trace"] == ["fake_tool"]
    assert any(isinstance(m, ToolMessage) for m in result["messages"])


def test_message_to_text_handles_gemini_block_content():
    from agents._helpers import message_to_text

    blocks = [
        {"type": "text", "text": "Part one"},
        {"type": "text", "text": "Part two"},
    ]

    assert message_to_text(blocks) == "Part one\nPart two"
    assert message_to_text("plain") == "plain"
    assert message_to_text([]) == ""
    assert message_to_text(None) == ""


def test_collect_tool_names_lists_unique_names_in_order():
    from agents._helpers import collect_tool_names

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {}, "id": "1", "type": "tool_call"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {}, "id": "2", "type": "tool_call"},
                {"name": "scale_pan_geometry", "args": {}, "id": "3", "type": "tool_call"},
            ],
        ),
        AIMessage(content="done"),
    ]

    assert collect_tool_names(messages) == ["web_search", "scale_pan_geometry"]
    assert collect_tool_names([]) == []


def test_parse_recipe_extracts_name_source_and_url():
    from agents.search_agent import _parse_recipe

    answer = """Here you go!

### Classic Chocolate Yogurt Cake
- **Source:** BBC Good Food - https://www.bbcgoodfood.com/recipes/12345
- **Prep Time:** 15 mins

#### Ingredients:
- 200g flour
"""
    recipe = _parse_recipe(answer)

    assert recipe["name"] == "Classic Chocolate Yogurt Cake"
    assert recipe["url"] == "https://www.bbcgoodfood.com/recipes/12345"
    assert "BBC Good Food" in recipe["source"]


def test_parse_recipe_returns_none_when_shape_is_unexpected():
    from agents.search_agent import _parse_recipe

    assert _parse_recipe("no recipe structure here") is None
    assert _parse_recipe("") is None


def test_search_prompt_selects_language_and_preserves_recipe_constraints():
    from agents.search_agent import build_system_prompt

    prompt = build_system_prompt("French")

    assert "respond entirely in French" in prompt
    assert "### [Recipe Name]" in prompt
    assert "[Missing - required]" in prompt
    assert "[Optional]" in prompt
    assert "[Quantity in cups/tbsp] ([Quantity in g/ml])" in prompt
    assert "Do not invent or hallucinate recipes" in prompt
    assert "quantities, units, temperatures, timings" in prompt


def test_search_agent_uses_language_from_state_and_defaults_to_english(monkeypatch):
    from agents import search_agent as search_module

    observed = []

    class FakeSearchModel:
        def invoke(self, messages):
            observed.append(messages[0].content)
            return AIMessage(content="No recipe found")

    monkeypatch.setattr(search_module, "search_agent", FakeSearchModel())
    search_module.search_agent_node({"messages": [], "language": "Arabic"})
    search_module.search_agent_node({"messages": []})

    assert "entirely in Arabic" in observed[0]
    assert "entirely in English" in observed[1]
