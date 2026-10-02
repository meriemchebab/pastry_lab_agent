"""Central model factory using LangChain's init_chat_model."""
from langchain.chat_models import init_chat_model


def init_gemini(temperature: float = 0.0):
    """
    Returns a ready-to-use Gemini chat model via the unified init_chat_model API.
   
    Args:
        temperature
    """
    return init_chat_model(
        model="gemini-3.6-flash",
        model_provider="google_genai",
        temperature=temperature,
    )
