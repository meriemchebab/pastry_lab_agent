"""Pan conversions, scaling, and substitution reasoning."""
from langchain_core.messages import SystemMessage
from ._model import init_gemini
from tools.baking_math import scale_pan_geometry, calculate_fat_substitution
from core.state import AgentBakingState

# Initialise model and bind tools once at import time
math_model = init_gemini(temperature=0.1)
math_agent = math_model.bind_tools([scale_pan_geometry, calculate_fat_substitution])

system_prompt = """
You are the Pastry Lab's Food Science & Math Agent. Your job is to help users scale recipes for different pan sizes or substitute missing ingredients based on fat/moisture chemistry.

### Core Instructions:
1. NEVER guess or estimate conversions. ALWAYS use your provided tools (`scale_pan_geometry` or `calculate_fat_substitution`).
2. When the tool returns the calculation, explain it simply to the user.
3. If a substitution requires adding or withholding liquid (water/milk), explicitly highlight this step so the user doesn't ruin their batter emulsion.
4. Keep your response focused entirely on the math and chemistry of the adjustment.
"""

def math_agent_node(state: AgentBakingState):
    """
    Executes when the user asks about pan sizes, scaling, or ingredient swaps.
    """
    messages = state.get("messages", [])
    conversation = [SystemMessage(content=system_prompt)] + messages

    # The model will invoke the tool, read the result, and generate the final answer
    response = math_agent.invoke(conversation)
    return {"messages": [response]}