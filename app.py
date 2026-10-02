import base64
import uuid
from typing import Any, cast
import gradio as gr
from gradio.themes import Soft
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

from core.graph import compiled_graph

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
            body_background_fill_dark="#2A2120",
            block_background_fill_dark="#1E1E1E",
            button_primary_background_fill_dark="#D94E86",
            button_primary_background_fill_hover_dark="#C73A73",
            button_secondary_background_fill_dark="#995F30",
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
        return gr.MultimodalTextbox(value=None), chat_history, gr.update(), gr.update()

    human_msg = HumanMessage(content=content)

    if text:
        chat_history.append({"role": "user", "content": text})
    for file_path in files:
        chat_history.append({"role": "user", "content": (file_path,)})

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    response = compiled_graph.invoke(
        cast(Any, {"messages": [human_msg]}), config=config
    )

    final_ai_msg = response["messages"][-1].content
    if isinstance(final_ai_msg, list):
        final_text = "\n".join(
            block["text"]
            for block in final_ai_msg
            if block.get("type") == "text"
        )
    else:
        final_text = str(final_ai_msg)

    chat_history.append({"role": "assistant", "content": final_text})

    state = compiled_graph.get_state(config).values
    active_recipe = state.get("active_recipe")

    recipe_display_text = (
        "Active Recipe Loaded" if active_recipe else "No recipe active."
    )
    step_display_text = f"Step {state.get('current_step', 1)}"

    return (
        gr.MultimodalTextbox(value=None),
        chat_history,
        recipe_display_text,
        step_display_text,
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
        outputs=[chat_input, chatbot, recipe_display, step_display],
    )

if __name__ == "__main__":
    demo.launch(theme=custom_theme)