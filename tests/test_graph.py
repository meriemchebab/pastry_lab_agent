"""Offline tests for LangGraph wiring and turn-to-turn state handling."""

import os
from dataclasses import replace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


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
