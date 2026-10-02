import os
from langchain_core.tools import tool
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> str:
    """
    Search the web for real recipes, cooking instructions, and ingredient substitutions. 
    Use specific queries like 'best chocolate cake recipe' or 'recipes using chicken'.
    """
    # search_depth="advanced"
    response = tavily_client.search(query, search_depth="advanced", max_results=3)
    
    # format for the llm to state the source : the URL and title
    formatted_results = []
    for i, result in enumerate(response.get("results", [])):
        title = result.get("title", "Unknown Title")
        url = result.get("url", "Unknown URL")
        content = result.get("content", "No content available.")
        
        formatted_results.append(
            f"--- Source {i+1} ---\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content Extract: {content}\n"
        )
        
    return "\n".join(formatted_results) if formatted_results else "No recipes found."
