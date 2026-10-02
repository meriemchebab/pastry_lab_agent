"""Intent classification and routing logic."""
from langchain_core.messages import SystemMessage
from ._model import init_gemini
from core.state import AgentBakingState

# Initialise model once at import time
model = init_gemini(temperature=0.0)

system_prompt = """
Classify the user's intent into ONE of the following categories:
- 'search': The user is asking for recipe ideas, stating ingredients they have, or looking for something to bake.
- 'math': The user is asking to scale a recipe, change pan sizes, or substitute an ingredient.
- 'copilot': The user is actively baking, asking for the next step, confirming they finished a step, or asking about kitchen timing.

Respond with EXACTLY ONE WORD: search, math, or copilot.
"""

def router_node(state: AgentBakingState):
    """Inspects the latest input and updates 'current_agent' in the state."""
    messages = state.get("messages", [])
    if not messages:
        return {"current_agent": "search"}

    last_msg = messages[-1]

    
    if isinstance(last_msg.content, list):
        for block in last_msg.content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                return {"current_agent": "vision"}

    
    text_content = (
        last_msg.content
        if isinstance(last_msg.content, str)
        else str(last_msg.content)
    )
    if "SYSTEM ALERT:" in text_content:
        return {"current_agent": "copilot"}

    # LLM-based intent classification
    response = model.invoke([SystemMessage(content=system_prompt), last_msg])
    response_content = response.content
    if isinstance(response_content, str):
        decision = response_content.strip().lower()
    else:
        text_parts = []
        for block in response_content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        decision = " ".join(text_parts).strip().lower()

    # Fallback in case of hallucination
    if decision not in {"search", "math", "copilot"}:
        decision = "search"

    return {"current_agent": decision}