"""Conversation memory compaction node."""
from langchain_core.messages import RemoveMessage, SystemMessage

from ._model import init_gemini
from core.state import AgentBakingState

model = init_gemini(temperature=0.0)


def summarize_memory_node(state: AgentBakingState):
    """Replace older conversation messages with a concise summary."""
    messages = state.get("messages", [])
    if len(messages) <= 10:
        return {}

    older_messages = messages[:-2]
    recent_messages = messages[-2:]
    response = model.invoke([
        SystemMessage(content=(
            "Summarize the conversation history below in exactly 3 sentences. "
            "Preserve recipe details, user preferences, and current progress."
        )),
        *older_messages,
    ])
    summary = response.content
    if not isinstance(summary, str):
        summary = " ".join(
            block.get("text", "") for block in summary
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )

    removals = [RemoveMessage(id=message.id) for message in older_messages if message.id]
    return {"messages": [*removals, SystemMessage(content=f"Conversation summary: {summary}")]}
