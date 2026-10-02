"""Single-step progression and sensory verification."""
from langchain_core.messages import SystemMessage
from ._model import init_gemini
from core.state import AgentBakingState

# Slightly higher temperature for a warm, encouraging tone
copilot_model = init_gemini(temperature=0.3)

def copilot_agent_node(state: AgentBakingState):
    """
    Executes when the user is actively baking and needs the next instruction.
    """
    messages = state.get("messages", [])

    # Get the step the user is on
    current_step = state.get("current_step", 1)

    system_prompt = f"""
    You are the Live Kitchen Co-Pilot. The user is actively baking a recipe found earlier in this conversation.

    ### Your Task:
    1. Read the chat history to find the recipe the user selected.
    2. The user is currently on **Step {current_step}**.
    3. Extract ONLY Step {current_step} from the recipe and present it to the user. DO NOT give them the next steps yet.
    4. Enhance the step with sensory cues. For example, instead of just "cream butter and sugar," add: "Look for the mixture to become noticeably pale, fluffy, and double in volume—usually takes about 3-4 minutes."
    5. End your message by asking the user to confirm they are ready for the next step (e.g., "Let me know when that's in the oven!" or "Tell me when it looks fluffy.").
    """

    conversation = [SystemMessage(content=system_prompt)] + messages
    response = copilot_model.invoke(conversation)

    return {
        "messages": [response],
        "current_step": current_step + 1,  # increment the step counter each call
    }