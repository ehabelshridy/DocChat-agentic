"""
graph.py

The LangGraph orchestration for DocChat, matching the confirmed diagram:

    start -> retrieve -> grade_relevance --[relevant]--> generate_answer
                              |                                |
                       [irrelevant]                    check_groundedness
                              v                          |            |
                       rewrite_query              [grounded]   [hallucinated]
                         |        |                    |              |
                  [retry]    [max retries]            end      (loop back to
                    |              |                                generate_answer,
                    v              v                                or [max retries]
                retrieve      fallback <----------------------------- to fallback)
                                   |
                                   v
                                  end

Two independent retry budgets:
  - relevance_retry_count: how many times we've rewritten the query
    after retrieval came back irrelevant.
  - generation_retry_count: how many times we've regenerated the
    answer after it failed the groundedness check.

Both loops terminate at the SAME fallback node once their budget is
exhausted, so the user always gets an honest answer (either a grounded
one, or a clear "I don't have reliable information for this") instead
of either an infinite loop or a hallucinated guess.
"""

from typing import List, TypedDict

from langgraph.graph import StateGraph, END

from llm import chat_completion, parse_json_response
from retrieval import HybridRetriever, RetrievedChunk

MAX_RELEVANCE_RETRIES = 2
MAX_GENERATION_RETRIES = 2
TOP_K = 5


class GraphState(TypedDict):
    question: str                      # original user question, never overwritten
    current_query: str                 # query actually sent to the retriever (may be rewritten)
    retrieved_chunks: List[RetrievedChunk]   # raw candidates from the retriever (may include noise)
    relevant_chunks: List[RetrievedChunk]    # subset that passed per-chunk relevance grading
    relevance_retry_count: int
    generation_retry_count: int
    is_relevant: bool
    answer: str
    is_grounded: bool
    fallback_reason: str               # "out_of_scope" | "unverified" | ""
    final_response: str                # what the API actually returns to the user


# A single retriever instance, reused across requests/graph runs.
# Built lazily so importing graph.py doesn't require chroma_db/ to
# exist yet (useful for tests).
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(top_k=TOP_K)
    return _retriever


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: GraphState) -> GraphState:
    query = state.get("current_query") or state["question"]
    chunks = get_retriever().retrieve(query, final_k=TOP_K)
    return {**state, "current_query": query, "retrieved_chunks": chunks}


RELEVANCE_SYSTEM_PROMPT = """You are a relevance grader for a document
QA system. Given a user question and ONE retrieved excerpt, decide
whether this excerpt alone contains information that helps answer the
question. Other excerpts from unrelated documents may be present
elsewhere in the candidate pool -- ignore them and judge ONLY this
excerpt on its own merits. If this excerpt is about a different
product/topic entirely, it is not relevant. If it directly addresses
the question's subject, it is relevant, even if it doesn't answer
every detail.

Respond with ONLY a JSON object, no other text:
{"is_relevant": true or false, "reasoning": "one short sentence"}"""


def grade_relevance_node(state: GraphState) -> GraphState:
    """Grades each retrieved chunk independently rather than judging
    the whole batch as one blob. This matters because hybrid retrieval
    intentionally returns several candidates -- some chunks WILL be
    irrelevant by design (that's the dense retriever casting a wide
    net). Judging them as a single combined context means a handful of
    off-topic chunks can drag down a genuinely relevant one in the
    judge's verdict. Per-chunk grading keeps the good chunk regardless
    of how much noise surrounds it.
    """
    relevant_chunks: List[RetrievedChunk] = []

    for chunk in state["retrieved_chunks"]:
        user_prompt = f"Question: {state['question']}\n\nExcerpt:\n{chunk['text']}"
        # temperature=0: this is a binary yes/no classification, not
        # creative generation. Any temperature above 0 lets a small
        # judge model flip its verdict on IDENTICAL input across calls
        # (observed in practice: the same chunk graded relevant in one
        # call and irrelevant in the next), which silently breaks the
        # retry logic below since retries assume the judge is at least
        # consistent for unchanged input.
        raw = chat_completion(RELEVANCE_SYSTEM_PROMPT, user_prompt, max_tokens=100, temperature=0.0)
        parsed = parse_json_response(raw)
        if bool(parsed.get("is_relevant", parsed.get("decision", False))):
            relevant_chunks.append(chunk)

    return {
        **state,
        "is_relevant": len(relevant_chunks) > 0,
        "relevant_chunks": relevant_chunks,
    }


REWRITE_SYSTEM_PROMPT = """You rewrite search queries for a document
retrieval system. The previous query did not retrieve relevant
results. Rewrite the user's question into a different search query
that is more likely to match the wording used in the source documents
(e.g. expand abbreviations, use clinical/technical synonyms, or
broaden an overly narrow phrase).

Respond with ONLY the rewritten query text, nothing else."""


def rewrite_query_node(state: GraphState) -> GraphState:
    user_prompt = f"Original question: {state['question']}\nPrevious search query: {state['current_query']}"
    new_query = chat_completion(REWRITE_SYSTEM_PROMPT, user_prompt, max_tokens=100, temperature=0.3)

    return {
        **state,
        "current_query": new_query.strip(),
        "relevance_retry_count": state["relevance_retry_count"] + 1,
    }


GENERATE_SYSTEM_PROMPT = """You are a careful assistant that answers
questions using ONLY the provided document excerpts. Rules:
1. Use ONLY information present in the excerpts below. Never use
   outside knowledge, even if you are confident it is correct.
2. If the excerpts do not fully answer the question, say so plainly
   instead of filling gaps with assumptions.
3. When relevant, mention which section the information came from.
4. Be concise and directly answer what was asked."""


def generate_answer_node(state: GraphState) -> GraphState:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source_file']} | Section: {c['headings']}]\n{c['text']}"
        for c in state["relevant_chunks"]
    )
    user_prompt = f"Document excerpts:\n{context}\n\nQuestion: {state['question']}"

    answer = chat_completion(GENERATE_SYSTEM_PROMPT, user_prompt, max_tokens=512, temperature=0.2)
    return {**state, "answer": answer}


GROUNDEDNESS_SYSTEM_PROMPT = """You are a strict fact-checker. Given a
set of document excerpts and a generated answer, determine whether
EVERY factual claim in the answer is directly supported by the
excerpts. Any invented detail, number, or claim not present in the
excerpts means the answer is NOT grounded, even if it sounds
plausible or correct in general.

Respond with ONLY a JSON object, no other text:
{"is_grounded": true or false, "reasoning": "one short sentence"}"""


def check_groundedness_node(state: GraphState) -> GraphState:
    context = "\n\n---\n\n".join(c["text"] for c in state["relevant_chunks"])
    user_prompt = f"Document excerpts:\n{context}\n\nGenerated answer:\n{state['answer']}"

    raw = chat_completion(GROUNDEDNESS_SYSTEM_PROMPT, user_prompt, max_tokens=150, temperature=0.0)
    print(f"\n[GROUNDEDNESS] attempt={state['generation_retry_count']+1} raw={raw!r}\n")
    parsed = parse_json_response(raw)
    is_grounded = bool(parsed.get("is_grounded", parsed.get("decision", False)))
    print(f"[GROUNDEDNESS] is_grounded={is_grounded}\n")

    updated = {**state, "is_grounded": is_grounded}
    if is_grounded:
        updated["final_response"] = state["answer"]
    else:
        updated["generation_retry_count"] = state["generation_retry_count"] + 1
    return updated


def fallback_node(state: GraphState) -> GraphState:
    if state.get("fallback_reason") == "out_of_scope" or not state.get("is_relevant", True):
        message = (
            "I couldn't find information about this in the documents I have access to. "
            "This may be outside the scope of the current knowledge base."
        )
        reason = "out_of_scope"
    else:
        message = (
            "I found related content, but couldn't generate an answer I could fully verify "
            "against the source documents, so I don't want to risk giving you inaccurate "
            "information. Could you rephrase the question or ask something more specific?"
        )
        reason = "unverified"

    return {**state, "final_response": message, "fallback_reason": reason}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_relevance(state: GraphState) -> str:
    if state["is_relevant"]:
        return "generate"
    if state["relevance_retry_count"] < MAX_RELEVANCE_RETRIES:
        return "rewrite"
    return "fallback_out_of_scope"


def route_after_rewrite(state: GraphState) -> str:
    # rewrite_query_node always loops back to retrieve; the "max retries"
    # exit is handled by route_after_relevance on the NEXT pass through
    # grade_relevance, so a query that still comes back irrelevant after
    # the budget is spent falls through to fallback there instead of here.
    return "retrieve"


def route_after_groundedness(state: GraphState) -> str:
    if state["is_grounded"]:
        return "end"
    if state["generation_retry_count"] < MAX_GENERATION_RETRIES:
        return "regenerate"
    return "fallback_unverified"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_relevance", grade_relevance_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("check_groundedness", check_groundedness_node)
    workflow.add_node("fallback", fallback_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_relevance")

    workflow.add_conditional_edges(
        "grade_relevance",
        route_after_relevance,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
            "fallback_out_of_scope": "fallback",
        },
    )

    workflow.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {"retrieve": "retrieve"},
    )

    workflow.add_edge("generate_answer", "check_groundedness")

    workflow.add_conditional_edges(
        "check_groundedness",
        route_after_groundedness,
        {
            "end": END,
            "regenerate": "generate_answer",
            "fallback_unverified": "fallback",
        },
    )

    workflow.add_edge("fallback", END)

    return workflow.compile()


def run_query(question: str) -> dict:
    """Entry point used by the FastAPI layer."""
    graph = build_graph()
    initial_state: GraphState = {
        "question": question,
        "current_query": question,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "relevance_retry_count": 0,
        "generation_retry_count": 0,
        "is_relevant": False,
        "answer": "",
        "is_grounded": False,
        "fallback_reason": "",
        "final_response": "",
    }
    final_state = graph.invoke(initial_state)
    return {
        "answer": final_state["final_response"],
        "sources": [
            {"source_file": c["source_file"], "section": c["headings"]}
            for c in final_state["relevant_chunks"]
        ] if final_state.get("is_grounded") else [],
        "retries": {
            "relevance": final_state["relevance_retry_count"],
            "generation": final_state["generation_retry_count"],
        },
    }
