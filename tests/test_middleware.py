"""Tests for telemetry, memory compaction, and recipe state reset."""
import os
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage


with patch.dict(os.environ, {"GOOGLE_API_KEY": "offline-test-key"}):
    from app import invoke_with_telemetry
    import agents.router as router_module
    import agents.summarizer as summarizer_module


def test_telemetry_rejects_input_over_token_limit_without_invoking(monkeypatch):
    def should_not_invoke(*args, **kwargs):
        raise AssertionError("oversized input should not invoke the graph")

    monkeypatch.setattr("app.compiled_graph.invoke", should_not_invoke)
    response, error = invoke_with_telemetry(HumanMessage(content="x" * 16004), "thread")

    assert response is None
    assert error is not None
    assert "4001 tokens" in error


def test_telemetry_invokes_and_prints_token_usage(monkeypatch, capsys):
    expected = {"messages": [AIMessage(content="Done", response_metadata={"token_usage": {"output_tokens": 3}})]}
    calls = []

    def fake_invoke(payload, **kwargs):
        calls.append((payload, kwargs))
        return expected

    monkeypatch.setattr("app.compiled_graph.invoke", fake_invoke)
    result, error = invoke_with_telemetry(HumanMessage(content="Hi"), "thread-1")

    assert error is None
    assert result is expected
    assert calls[0][1]["config"]["configurable"]["thread_id"] == "thread-1"
    assert "output_tokens" in capsys.readouterr().out


def test_summarizer_leaves_ten_or_fewer_messages_untouched(monkeypatch):
    def should_not_invoke(*args, **kwargs):
        raise AssertionError("short history should not be summarized")

    class NoCallModel:
        def invoke(self, *args, **kwargs):
            should_not_invoke(*args, **kwargs)

    monkeypatch.setattr(summarizer_module, "model", NoCallModel())
    messages = [HumanMessage(content=f"message {i}") for i in range(10)]

    assert summarizer_module.summarize_memory_node({"messages": messages}) == {}


def test_summarizer_summarizes_old_messages_and_keeps_two_recent(monkeypatch):
    messages = [HumanMessage(content=f"message {i}", id=f"message-{i}") for i in range(11)]
    class SummaryModel:
        def invoke(self, prompt):
            return AIMessage(content="First sentence. Second sentence. Third sentence.")

    monkeypatch.setattr(summarizer_module, "model", SummaryModel())

    update = summarizer_module.summarize_memory_node({"messages": messages})
    entries = update["messages"]

    assert all(isinstance(item, RemoveMessage) for item in entries[:-1])
    assert [item.id for item in entries[:-1]] == [item.id for item in messages[:-2]]
    assert isinstance(entries[-1], SystemMessage)
    assert "First sentence. Second sentence. Third sentence." in entries[-1].content


def test_router_search_resets_active_recipe_and_step(monkeypatch):
    class SearchModel:
        def invoke(self, prompt):
            return AIMessage(content="search")

    monkeypatch.setattr(router_module, "model", SearchModel())
    state = {
        "messages": [HumanMessage(content="Find another recipe")],
        "active_recipe": {"name": "Old recipe"},
        "current_step": 4,
    }

    result = router_module.router_node(state)

    assert result == {"current_agent": "search", "active_recipe": None, "current_step": 1}


def test_router_non_search_intent_preserves_recipe_state(monkeypatch):
    class MathModel:
        def invoke(self, prompt):
            return AIMessage(content="math")

    monkeypatch.setattr(router_module, "model", MathModel())
    state = {
        "messages": [HumanMessage(content="Scale it")],
        "active_recipe": {"name": "Cake"},
        "current_step": 3,
    }

    result = router_module.router_node(state)

    assert result == {"current_agent": "math"}
