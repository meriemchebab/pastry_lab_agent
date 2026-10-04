# The Pastry Lab

The Pastry Lab is a stateful, multi-agent baking assistant built with LangGraph, Google Gemini (llm model), and Gradio (UI). The main agent lets the user interact with specialized agents that handle web recipe searches based on the user available ingredients, image-based pantry detection, deterministic baking chemistry and pan scaling calculations, and step-by-step baking guidance.

## Features

- **Recipe search:** Find real online recipes based on ingredients or baking ideas, with source links.
- **Ingredient photo analysis:** Upload a pantry or countertop photo to identify visible ingredients and possible gaps.
- **Baking math:** Scale a recipe between round and square pans by surface area using inches or centimeters, and calculate fat and moisture adjustments for supported ingredient substitutions.
- **Guided baking:** Get the current recipe step with practical sensory cues, then continue as you bake.
- **Conversation state:** Keep recipe and baking progress within a chat session.
- **Response language:** Choose English, Arabic, or French in the Settings sidebar. Recipe search responses follow the selected language while preserving recipe formatting, ingredient status tags, and exact measurements.

## Requirements

- Python 3.13 or later
- A Google AI API key for Gemini
- A Tavily API key for recipe search

## Setup

Clone or download this project, then open a terminal in its directory.

Create and activate a virtual environment:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:

```dotenv
GOOGLE_API_KEY=your_google_ai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The app loads `.env` on startup. Keep your real keys private and do not commit them.

## Run the app

```powershell
python app.py
```

Open the local Gradio URL printed in the terminal. Choose a response language in the Settings sidebar, then enter a baking request in the chat box or attach an image to ask about visible ingredients. The language preference is passed into graph state for the request; the search agent uses it to translate its full recipe response while retaining source names and URLs, the Master Baker output format, missing ingredient tags, and recipe quantities and units.

## How requests are handled

The LangGraph workflow routes each message to a specialist:

| Request | Specialist | Tools |
| --- | --- | --- |
| Recipe ideas or recipes from available ingredients | Search agent | Tavily web search |
| Pan-size scaling or supported fat substitutions | Math agent | Pan geometry and ingredient composition calculations |
| An uploaded ingredient photo | Vision agent | Gemini image analysis |
| Help while following a recipe | Kitchen copilot | Recipe and conversation context |

The search and math agents can call their tools in a loop before returning an answer. Conversation state is kept in memory for the lifetime of the app process and is scoped to a chat session.

## Supported substitution ingredients

Use these ingredient keys when asking for substitutions:

`butter_standard`, `neutral_oil`, `water`, `whole_milk`, `whole_egg`, and `granulated_sugar`.

Substitution math matches fat content and reports how much liquid to add or withhold to balance moisture. Pan scaling accepts `round` or `square` pans and dimensions in inches or centimeters (centimeters are assumed when no unit is specified).

## Run tests

The test suite exercises baking calculations, imports, and graph behavior without making live API calls:

```powershell
pytest
```

## Project layout

```text
app.py                  Gradio interface and request handling
agents/                 Intent router and specialist agents
core/                   LangGraph workflow and shared state
tools/                  Recipe search and baking calculations
tests/                  Offline unit and workflow tests
```
## License

This project is licensed under the [MIT License](LICENSE).
