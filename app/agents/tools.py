import json

from app.core.chroma_store import ChromaStore

SEARCH_DOCUMENTS = "search_documents"
LIST_DOCUMENTS = "list_documents"

# Offered to the planner through the LLM's native tool-calling API; the model decides which to call.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": SEARCH_DOCUMENTS,
            "description": "Semantic search over the content of the uploaded documents. "
                           "Use for any question that could be answered from the documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise, keyword-rich search phrase. Keep technical terms and expand abbreviations.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": LIST_DOCUMENTS,
            "description": "List the file names of the indexed documents and the total chunk count. "
                           "Use when the user asks what documents are available.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(name: str, store: ChromaStore) -> str:
    """Run a tool whose result goes straight back to the planner.

    search_documents is not handled here: it ends planning and is executed by the
    workflow's retrieve node, which adds the score threshold and wider retry search.
    """
    if name == LIST_DOCUMENTS:
        return json.dumps({"documents": store.list_sources(), "chunks": store.count()})
    return json.dumps({"error": f"Unknown tool: {name}"})
