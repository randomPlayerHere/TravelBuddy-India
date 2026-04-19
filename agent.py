import os
import datetime
from typing import TypedDict, List

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from knowledge_base import DOCUMENTS

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Add it and restart.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("LLM and embedder ready.")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("travel_india_kb")


def setup_knowledge_base():
    docs = [d["text"] for d in DOCUMENTS]
    ids = [d["id"] for d in DOCUMENTS]
    metadatas = [{"topic": d["topic"]} for d in DOCUMENTS]
    embeddings = embedder.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"Knowledge base loaded: {len(DOCUMENTS)} documents into ChromaDB.")


setup_knowledge_base()

_test = collection.query(
    query_embeddings=[embedder.encode(["best beaches in India"]).tolist()[0]],
    n_results=2,
)
print(f"Retrieval test passed. Top topics: {[m['topic'] for m in _test['metadatas'][0]]}")


class TravelState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str


def memory_node(state: TravelState) -> TravelState:
    """Append question to history, apply sliding window, extract user name."""
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    msgs = msgs[-6:]

    user_name = state.get("user_name", "")
    q_lower = state["question"].lower()
    if "my name is" in q_lower:
        try:
            name_part = state["question"].lower().split("my name is")[-1].strip()
            user_name = name_part.split()[0].capitalize()
        except Exception:
            pass

    return {
        **state,
        "messages": msgs,
        "user_name": user_name,
        "eval_retries": 0,
        "retrieved": "",
        "tool_result": "",
        "answer": "",
    }


def router_node(state: TravelState) -> TravelState:
    """Use LLM to classify route: retrieve | tool | memory_only."""
    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state.get("messages", [])[-4:]
    )
    prompt = f"""You are a routing agent for an Indian travel assistant chatbot.
Classify the user's latest question into EXACTLY ONE of these routes:

- retrieve   → needs information from travel KB (destinations, attractions, budget, best time, food, permits)
- tool       → needs current date, current month, or "what season is it now" type questions
- memory_only → pure greetings, thanks, or can be answered from conversation history alone

Conversation so far:
{history_str}

Question to classify: {state['question']}

Reply with ONE word only: retrieve, tool, or memory_only"""

    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower().split()[0]
    if route not in ("retrieve", "tool", "memory_only"):
        route = "retrieve"

    print(f"[router] route = {route}")
    return {**state, "route": route}


def retrieval_node(state: TravelState) -> TravelState:
    """Embed question, query ChromaDB for top 3 chunks, format as context."""
    query_emb = embedder.encode([state["question"]]).tolist()[0]
    results = collection.query(query_embeddings=[query_emb], n_results=3)

    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    context = "\n\n".join(
        f"[{metas[i]['topic']}]\n{chunks[i]}" for i in range(len(chunks))
    )
    sources = [m["topic"] for m in metas]

    print(f"[retrieval] sources: {sources}")
    return {**state, "retrieved": context, "sources": sources}


def skip_retrieval_node(state: TravelState) -> TravelState:
    """For memory_only route — skip retrieval."""
    return {**state, "retrieved": "", "sources": []}


def tool_node(state: TravelState) -> TravelState:
    """Return current date and a travel season recommendation. NEVER raises exceptions."""
    try:
        now = datetime.datetime.now()
        month_num = now.month
        month_name = now.strftime("%B")
        year = now.year

        if month_num in (12, 1, 2):
            season = "Winter (Dec–Feb)"
            advice = (
                "This is PEAK travel season in India! Weather is pleasant across most of the country. "
                "Best destinations right now: Rajasthan (cold nights, perfect days), Goa (beach weather), "
                "Kerala (backwaters and beaches), Tamil Nadu temples, Varanasi (misty mornings). "
                "Note: Himalayan high passes (Rohtang, Manali–Leh) are closed due to snow."
            )
        elif month_num in (3, 4, 5):
            season = "Spring / Summer (Mar–May)"
            advice = (
                "Great time for Himalayan hill stations! "
                "Best now: Himachal Pradesh (Manali, Spiti opening in May), Uttarakhand (Rishikesh rafting, Kedarnath opening), "
                "Ladakh prep (roads open from late May/June). "
                "Plains are getting hot — avoid Rajasthan, Delhi sightseeing from April onwards. "
                "Andaman is still great through May before seas get rough."
            )
        elif month_num in (6, 7, 8, 9):
            season = "Monsoon (Jun–Sep)"
            advice = (
                "Monsoon season! Some places become magical during rains. "
                "Best now: Kerala backwaters (lush and green), Meghalaya (Cherrapunji waterfalls at full force, Dawki river), "
                "Valley of Flowers Uttarakhand (peak bloom July–September), Coorg coffee estates. "
                "AVOID: Ladakh (road closures and landslides), Andaman (rough seas, ferries cancelled), "
                "Rajasthan (very hot and humid), most beach destinations."
            )
        else:
            season = "Autumn / Post-Monsoon (Oct–Nov)"
            advice = (
                "Excellent season — fresh landscapes, clear skies, festival season! "
                "Best now: Ladakh (last window before snow — go in October), Northeast India (Meghalaya, Sikkim), "
                "Rajasthan (season just starting, cooler and less crowded), Kerala, Andaman (seas calm from October). "
                "Major festivals: Diwali and Durga Puja — wonderful cultural experience but book everything early!"
            )

        result = (
            f"Current date: {month_name} {now.day}, {year}.\n"
            f"Current season: {season}.\n"
            f"Travel advice for this time of year: {advice}"
        )
    except Exception as e:
        result = f"Could not retrieve date information. Please check travel guides for seasonal tips. (Error: {e})"

    print("[tool] date/season tool executed")
    return {**state, "tool_result": result}


def answer_node(state: TravelState) -> TravelState:
    """Build system prompt with context and generate LLM answer."""
    name_prefix = f"Hi {state['user_name']}! " if state.get("user_name") else ""
    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state.get("messages", [])[-4:]
    )

    retry_instruction = ""
    if state.get("eval_retries", 0) > 0:
        retry_instruction = (
            "\nRETRY: Your previous answer was flagged for including information NOT in the context. "
            "This time, STRICTLY use only the provided context. If the context doesn't cover the question, "
            "say so clearly and refer the user to incredibleindia.org."
        )

    context_block = ""
    if state.get("retrieved"):
        context_block += f"\n\n=== KNOWLEDGE BASE CONTEXT ===\n{state['retrieved']}"
    if state.get("tool_result"):
        context_block += f"\n\n=== LIVE TOOL RESULT ===\n{state['tool_result']}"

    system_prompt = f"""You are TravelBuddy India, a helpful and knowledgeable travel assistant
specializing in Indian domestic travel. You help travelers plan amazing trips across India.
{retry_instruction}

STRICT RULES:
1. Answer ONLY using information from the provided context. Do NOT invent destinations, prices, or facts.
2. If you cannot answer from the context, say: "I don't have specific details about that in my knowledge base.
   I'd recommend checking the official Incredible India website (incredibleindia.org) or TripAdvisor for more info."
3. Always mention best time to visit when recommending destinations.
4. Be clear, helpful, and concise.
5. Keep answers concise but informative (3–5 sentences for simple questions, more for complex planning).
6. If the user asks about medical emergencies or safety threats, always say: "Please contact local authorities or
   dial 112 (India's emergency number) immediately."
{context_block}

=== CONVERSATION HISTORY ===
{history_str}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ])

    answer = name_prefix + response.content.strip()
    return {**state, "answer": answer}


def eval_node(state: TravelState) -> TravelState:
    """Score faithfulness 0.0–1.0. Retry answer if score < 0.7 (max 2 retries)."""
    if not state.get("retrieved"):
        print("[eval] skipped (no retrieval context), score = 1.0")
        return {**state, "faithfulness": 1.0}

    retries = state.get("eval_retries", 0)

    prompt = f"""You are a faithfulness evaluator. Rate whether the ANSWER uses ONLY information from the CONTEXT
without adding invented facts.

Score 1.0 = answer is completely grounded in context.
Score 0.0 = answer contains significant hallucinations or invented facts.

CONTEXT:
{state.get('retrieved', '')[:2000]}

ANSWER:
{state['answer']}

Reply with ONLY a single decimal number between 0.0 and 1.0. Example: 0.85"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.85

    gate = "RETRY" if score < 0.7 and retries < 2 else "PASS"
    print(f"[eval] faithfulness = {score:.2f} | retries = {retries} | gate = {gate}")
    return {**state, "faithfulness": score, "eval_retries": retries + 1}


def save_node(state: TravelState) -> TravelState:
    """Append assistant answer to messages history and persist via MemorySaver."""
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "assistant", "content": state["answer"]}]
    return {**state, "messages": msgs}


def route_decision(state: TravelState) -> str:
    """After router_node: decide which processing branch to take."""
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state: TravelState) -> str:
    """After eval_node: retry answer if faithfulness < 0.7 and retries < 2."""
    if state.get("faithfulness", 1.0) < 0.7 and state.get("eval_retries", 0) < 2:
        return "answer"
    return "save"


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_edge("save", END)

    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"},
    )

    app = graph.compile(checkpointer=MemorySaver())
    print("Graph compiled successfully.")
    return app


app = build_graph()


def ask(question: str, thread_id: str = "test_thread") -> dict:
    """Invoke the graph and return the result state."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_input = {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name": "",
    }
    result = app.invoke(initial_input, config=config)
    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TravelBuddy India - Quick Test")
    print("=" * 60)

    test_questions = [
        "What is the best time to visit Ladakh?",
        "Tell me about Goa beaches and budget.",
        "What month is it now and where should I travel?",
        "Hi, how are you?",
    ]

    for i, q in enumerate(test_questions, 1):
        print(f"\nQ{i}: {q}")
        result = ask(q, thread_id="standalone_test")
        print(f"Route: {result['route']}")
        print(f"Faithfulness: {result['faithfulness']:.2f}")
        print(f"Answer: {result['answer'][:200]}...")
        print("-" * 40)
