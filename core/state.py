# Graph state definitions
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import add_messages

class AgentBakingState(TypedDict):
    """
    The shared state dictionary that flows through all agents in the LangGraph application.
    """

    messages: Annotated[list, add_messages]
    current_agent: str
    active_recipe: Optional[Dict[str, Any]]
    current_step: int