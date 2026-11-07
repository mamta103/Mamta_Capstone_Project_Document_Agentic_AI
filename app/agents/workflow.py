from langgraph.graph import END, START, StateGraph

from app.agents.answer_agent import INSUFFICIENT_MESSAGE, generate, retrieve
from app.agents.planner import plan
from app.agents.safety import moderate_input, moderate_output
from app.agents.state import AgentState
from app.agents.validator import validate
from app.core.retriever import Retriever
from app.core.llm_client import LLMClient
from app.core.chroma_store import ChromaStore

MAX_ATTEMPTS = 2
RECURSION_LIMIT = 25


def route_after_input_check(state: AgentState) -> str:
    return "end" if state.get("status") == "blocked" else "planner"


def route_after_planner(state: AgentState) -> str:
    return state["route"]


def route_after_retrieve(state: AgentState) -> str:
    return "generate" if state["retrieved"] else "end"


def route_after_generate(state: AgentState) -> str:
    return "end" if state.get("status") in ("error", "insufficient_evidence") else "validate"


def route_after_validation(state: AgentState) -> str:
    v = state["validation"]
    if "error" in v:
        return "accept"
    if v["supported"] and v["complete"]:
        return "accept"
    if state["attempts"] < MAX_ATTEMPTS:
        return "retry"
    return "accept" if v["supported"] else "refuse"


def prepare_retry(state: AgentState) -> dict:
    v = state["validation"]
    parts = []
    if v["unsupported_claims"]:
        parts.append("Unsupported or invalid: " + "; ".join(v["unsupported_claims"]))
    if not v["complete"] and v["missing"]:
        parts.append("Missing information: " + v["missing"])
    feedback = "\n".join(parts) or "Improve grounding and citations."
    return {"feedback": feedback, "trace": ["Decision: retry with wider retrieval and validator feedback"]}


def accept(state: AgentState) -> dict:
    v = state["validation"]
    if "error" in v:
        note = "accepted WITHOUT validation (validator unavailable)"
    elif v["complete"]:
        note = "accepted (validated)"
    else:
        note = "accepted (supported but incomplete after retry)"
    return {"status": "answered", "trace": [f"Decision: answer {note}"]}


def refuse(state: AgentState) -> dict:
    return {
        "status": "insufficient_evidence",
        "answer": INSUFFICIENT_MESSAGE,
        "trace": ["Decision: answer still unsupported after retry -> refuse"],
    }


def build_workflow(llm: LLMClient, retriever: Retriever, store: ChromaStore):
    graph = StateGraph(AgentState)
    graph.add_node("moderate_input", lambda s: moderate_input(s, llm=llm, store=store))
    graph.add_node("planner", lambda s: plan(s, llm=llm, store=store))
    graph.add_node("retrieve", lambda s: retrieve(s, retriever=retriever))
    graph.add_node("generate", lambda s: generate(s, llm=llm))
    graph.add_node("validate", lambda s: validate(s, llm=llm))
    graph.add_node("prepare_retry", prepare_retry)
    graph.add_node("accept", accept)
    graph.add_node("refuse", refuse)
    graph.add_node("moderate_output", lambda s: moderate_output(s, llm=llm))

    graph.add_edge(START, "moderate_input")
    graph.add_conditional_edges("moderate_input", route_after_input_check, {"planner": "planner", "end": END})
    # Only LLM-written text passes through moderate_output; fixed system messages go straight to END.
    graph.add_conditional_edges(
        "planner", route_after_planner, {"retrieve": "retrieve", "respond": "moderate_output", "end": END}
    )
    graph.add_conditional_edges("retrieve", route_after_retrieve, {"generate": "generate", "end": END})
    graph.add_conditional_edges("generate", route_after_generate, {"validate": "validate", "end": END})
    graph.add_conditional_edges(
        "validate", route_after_validation, {"accept": "accept", "retry": "prepare_retry", "refuse": "refuse"}
    )
    graph.add_edge("prepare_retry", "retrieve")
    graph.add_edge("accept", "moderate_output")
    graph.add_edge("refuse", END)
    graph.add_edge("moderate_output", END)
    return graph.compile()


def run_workflow(workflow, question: str) -> AgentState:
    return workflow.invoke({"question": question, "trace": []}, config={"recursion_limit": RECURSION_LIMIT})
