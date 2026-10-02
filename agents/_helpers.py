"""Small shared helpers for reading model output and tool usage."""
from langchain_core.messages import AIMessage


def message_to_text(content) -> str:
    """Flatten Gemini's block-list content into plain text.

    Gemini returns `[{'type': 'text', 'text': ...}]`; other providers return a
    plain string. Both need to become text before they reach the UI.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def collect_tool_names(messages: list) -> list[str]:
    """Every distinct tool the model has asked for, oldest first.

    This is the single most useful thing to log when an answer comes back
    empty: if this is non-empty but the reply is blank, the tool call was
    never executed (or never looped back into the model).
    """
    names: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                name = call.get("name")
                if name and name not in names:
                    names.append(name)
    return names
