"""Recipe search – finds real recipes via the web."""
import re

from langchain_core.messages import SystemMessage
from ._helpers import collect_tool_names, message_to_text
from ._model import init_gemini
from tools.search_tool import web_search
from core.state import AgentBakingState


search_model = init_gemini(temperature=0.2)
search_agent = search_model.bind_tools([web_search])

_BASE_SYSTEM_PROMPT = """
You are an expert pastry chef and master baker with 15 years of professional experience in high-end bakeries. Your role is to suggest realistic, tested dessert and cake recipes based strictly or primarily on the ingredients the user has on hand.

### Core Instructions:
1. ALWAYS use the `web_search` tool to find real recipes on the internet based on the user's provided ingredients. Do not invent or hallucinate recipes.
2. Input Handling: Analyze the user's provided ingredients. Assume they have basic pantry staples (water, salt, pinch of sugar) only if strictly necessary.
3. Selection: Suggest between 1 and 3 tested, high-quality dessert or cake recipes that best match the provided ingredients.
4. Source Citation: You MUST provide the original URL for every recipe so the user can read the full post. Use the URLs provided by your search tool.
5. Tone: Encouraging, precise, and practical, focusing on pastry fundamentals.

### Output Format:
For each suggested dessert (up to 3), structure the response exactly using the following template:

### [Recipe Name]
- **Source:** [Website Name] - [Direct URL]
- **Prep Time:** [X mins] | **Bake/Cook Time:** [X mins] | **Yield:** [Servings/Pan Size]
- **Match Level:** [Exact Match / Requires 1-2 Common Pantry Items]

#### Ingredients:
- [Quantity in cups/tbsp] ([Quantity in g/ml]) [Ingredient Name]
*(If an ingredient is not in the user's list, append: `[Missing - required]` or `[Optional]`)*

#### Step-by-Step Instructions:
1. **[Step Name/Phase]:** [Clear, detailed instruction including target texture, pan prep, and temperatures]
2. **[Step Name/Phase]:** ...

#### Chef's Tip:
- [A 1-sentence tip on technique, avoiding common mistakes, or easy substitutes]
"""


def build_system_prompt(language: str = "English") -> str:
    """Add language direction without weakening recipe format and precision rules."""
    language = language or "English"
    return f"""You must translate and respond entirely in {language}. Translate recipe names, headings, labels, ingredient names, instructions, and chef's tips into {language}. Preserve source website names and URLs exactly as returned by search. Translation must not change quantities, units, temperatures, timings, ingredient status, or procedural meaning. Strictly maintain all Master Baker formatting rules below, including the template, missing ingredient tags, and precise measurements.

{_BASE_SYSTEM_PROMPT}"""

def _parse_recipe(text: str) -> dict | None:
    """Pull the first recipe out of the formatted answer.

    The system prompt pins the output shape, so this is cheaper and far more
    reliable than a second LLM call to summarise what we just wrote.
    """
    if not text:
        return None

    name_match = re.search(r"^###\s+(.+?)\s*$", text, re.MULTILINE)
    if not name_match:
        return None

    url_match = re.search(r"https?://\S+", text)
    source_match = re.search(r"\*\*Source:\*\*\s*(.+?)\s*$", text, re.MULTILINE)

    return {
        "name": name_match.group(1).strip(),
        "source": source_match.group(1).strip() if source_match else "Unknown",
        "url": url_match.group(0) if url_match else None,
    }


def search_agent_node(state: AgentBakingState):
    """
    Node that LangGraph will execute when routing to the Search Agent.

    On the first pass the model requests `web_search`; LangGraph routes to the
    `search_tools` node, then sends us back here with the results in `messages`
    and we produce the final, human-readable answer.
    """
    messages = state.get("messages", [])
    language = state.get("language", "English") or "English"
    system_prompt = build_system_prompt(language)
    conversation = [SystemMessage(content=system_prompt)] + messages
    response = search_agent.invoke(conversation)

    # Only the *final* pass (no pending tool calls) becomes the active recipe.
    if response.tool_calls:
        return {"messages": [response]}

    recipe = _parse_recipe(message_to_text(response.content))

    return {
        "messages": [response],
        "tool_trace": collect_tool_names(messages + [response]),
        "active_recipe": recipe or state.get("active_recipe"),
    }
