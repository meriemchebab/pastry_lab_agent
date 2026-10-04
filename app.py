import base64
import traceback
import uuid
from typing import Any, cast
import gradio as gr
from gradio.themes import Soft
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

from agents._helpers import message_to_text
from core.graph import compiled_graph

MAX_INPUT_TOKENS = 4000

def invoke_with_telemetry(user_message: HumanMessage, thread_id: str) -> tuple[dict | None, str | None]:
    """Invoke the graph for a single user message with an input-size guard."""
    text_payload = message_to_text(user_message.content)
    estimated_tokens = len(text_payload) // 4
    if estimated_tokens > MAX_INPUT_TOKENS:
        return None, (
            f"Your message is estimated at {estimated_tokens} tokens; "
            f"the limit is {MAX_INPUT_TOKENS}. Please shorten it and try again."
        )

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    response = compiled_graph.invoke(
        cast(Any, {"messages": [user_message]}),
        config=config,
        recursion_limit=25,
    )
    final_ai_msg = response["messages"][-1]
    usage = getattr(final_ai_msg, "usage_metadata", None)
    if not usage:
        usage = getattr(final_ai_msg, "response_metadata", {}).get("token_usage")
    print(f"Token usage: {usage if usage is not None else 'unavailable'}")
    return response, None

#  theme for gradio
class PastryTheme(Soft):
    def __init__(self):
        super().__init__()
        # Light‑mode
        self.set(
            body_background_fill="#E8E0D3",
            block_background_fill="#FFFFFF",
            block_title_text_color="#EC7CAA",
            button_primary_background_fill="#FF5A9B",          # <-- real colour
            button_primary_background_fill_hover="#F0A4C4",
            button_primary_text_color="white",
            button_secondary_background_fill="#FCC761",
            button_secondary_background_fill_hover="#fddb95",
            button_secondary_text_color="black",
            border_color_primary="#FF0BAC",
        )
        # Dark‑mode 
        self.set(
            body_background_fill_dark="#12013B",
            block_background_fill_dark="#312b6e",
            button_primary_background_fill_dark="#D94E86",
            button_primary_background_fill_hover_dark="#C73A73",
            button_secondary_background_fill_dark="#FCC761",
            button_secondary_background_fill_hover_dark="#99734F",
            border_color_primary_dark="#B04D8A",
        )
custom_theme = PastryTheme()

def encode_image(image_path: str) -> str:
    """Convert a Gradio‑uploaded image to a base‑64 string for Gemini."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def process_input(message_dict: dict, chat_history: list, thread_id: str):
    text = message_dict.get("text", "")
    files = message_dict.get("files", [])

    content = []
    if text:
        content.append({"type": "text", "text": text})

    for file_path in files:
        b64_data = encode_image(file_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
        })

    if not content:
        return gr.MultimodalTextbox(value=None), chat_history, gr.update(), gr.update(), gr.update()

    human_msg = HumanMessage(content=content)

    if text:
        chat_history.append({"role": "user", "content": text})
    for file_path in files:
        chat_history.append({"role": "user", "content": (file_path,)})

    # Surfaces failures in the chat instead of only in the terminal.
    try:
        response, size_error = invoke_with_telemetry(human_msg, thread_id)
    except Exception as exc:
        traceback.print_exc()
        chat_history.append({
            "role": "assistant",
            "content": f"**Something broke while I was working.**\n\n```\n{type(exc).__name__}: {exc}\n```",
        })
        return gr.MultimodalTextbox(value=None), chat_history, gr.update(), gr.update(), gr.update()

    if size_error:
        chat_history.append({"role": "assistant", "content": size_error})
        return gr.MultimodalTextbox(value=None), chat_history, gr.update(), gr.update(), gr.update()

    final_ai_msg = response["messages"][-1]
    final_text = message_to_text(final_ai_msg.content)

    # Blank reply with pending tool calls means the tool loop never resolved.
    if not final_text.strip():
        pending = ", ".join(c.get("name", "?") for c in (final_ai_msg.tool_calls or []))
        final_text = (
            f"**Empty reply.** The model asked for `{pending or 'no tool'}` "
            "but no result came back. Check the traceback above."
        )

    chat_history.append({"role": "assistant", "content": final_text})

    state = compiled_graph.get_state(config).values
    active_recipe = state.get("active_recipe")
    tool_trace = state.get("tool_trace") or []

    recipe_display_text = (
        f"{active_recipe['name']}" if active_recipe else "No recipe active."
    )
    step_display_text = f"Step {state.get('current_step', 1)}"
    tool_display_text = ", ".join(tool_trace) if tool_trace else "None yet"

    return (
        gr.MultimodalTextbox(value=None),
        chat_history,
        recipe_display_text,
        step_display_text,
        tool_display_text,
    )

# --- Gradio UI Layout ---
with gr.Blocks(title="The Pastry Lab", theme=custom_theme) as demo:
    thread_id = gr.State(value=lambda: str(uuid.uuid4()))

    gr.Markdown("# The Pastry Lab: Multi-Agent Co-Pilot")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Session State")
            recipe_display = gr.Textbox(
                label="Active Recipe",
                value="No recipe active.",
                interactive=False,
            )
            step_display = gr.Textbox(
                label="Current Progress",
                value="Step 1",
                interactive=False,
            )
            tool_display = gr.Textbox(
                label="Tools Used",
                value="None yet",
                interactive=False,
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Kitchen Copilot",
                
                height=600,
            )

            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_types=["image"],
                placeholder="Ask for a recipe, upload a photo of your pantry, or request a math conversion...",
                show_label=False,
            )

    chat_input.submit(
        fn=process_input,
        inputs=[chat_input, chatbot, thread_id],
        outputs=[chat_input, chatbot, recipe_display, step_display, tool_display],
    )

if __name__ == "__main__":
    demo.launch(theme=custom_theme,share=True)
