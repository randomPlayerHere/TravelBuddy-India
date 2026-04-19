import uuid
import streamlit as st

st.set_page_config(
    page_title="TravelBuddy India",
    page_icon="TB",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_agent():
    """Load LLM, embedder, ChromaDB, and compiled graph ONCE."""
    from agent import app
    return app


travel_app = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

with st.sidebar:
    st.title("TravelBuddy India")
    st.caption("Travel planning assistant for Indian destinations")
    st.divider()

    st.markdown("**Topics covered:**")
    topics = [
        "Rajasthan (forts, deserts, heritage)",
        "Kerala (backwaters, Munnar, beaches)",
        "Goa (beaches, nightlife, heritage)",
        "Himachal Pradesh (Manali, Shimla, Spiti)",
        "Ladakh (high passes, monasteries)",
        "Andaman Islands (coral reefs, diving)",
        "Varanasi and spiritual tourism",
        "Uttarakhand (trekking, Char Dham)",
        "Northeast India (Meghalaya, Sikkim)",
        "Tamil Nadu (Dravidian temples)",
        "Budget travel tips and apps",
        "Seasonal and monsoon travel guide",
    ]
    for t in topics:
        st.markdown(f"  {t}")

    st.divider()

    if st.button("New Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.user_name = ""
        st.rerun()

    st.divider()
    st.caption(f"Session ID: `{st.session_state.thread_id[:12]}...`")
    st.caption("Built with LangGraph · ChromaDB · Gemini · Streamlit")

st.title("TravelBuddy India")
st.markdown(
    "_Ask me about Indian destinations, best times to visit, travel budgets, "
    "seasonal tips, trekking, food, and more!_"
)

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Best time to visit Ladakh?"):
            st.session_state._inject = "What is the best time to visit Ladakh and what permits do I need?"
    with col2:
        if st.button("Where to go right now?"):
            st.session_state._inject = "What month is it currently and where should I travel in India right now?"
    with col3:
        if st.button("Budget trip to Goa?"):
            st.session_state._inject = "How do I plan a budget trip to Goa? What's the best time and how much will it cost?"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources consulted"):
                for src in msg["sources"]:
                    st.markdown(f"• **{src}**")
        if msg.get("faithfulness") is not None and msg["role"] == "assistant":
            score = msg["faithfulness"]
            color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
            st.caption(f"Faithfulness score: :{color}[{score:.2f}]")

prompt = None
if hasattr(st.session_state, "_inject"):
    prompt = st.session_state._inject
    del st.session_state._inject

user_input = st.chat_input("Ask about any Indian destination or travel tip...")
if user_input:
    prompt = user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Preparing response..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            from agent import TravelState
            state_input = TravelState(
                question=prompt,
                messages=st.session_state.messages[:-1],
                route="",
                retrieved="",
                sources=[],
                tool_result="",
                answer="",
                faithfulness=0.0,
                eval_retries=0,
                user_name=st.session_state.user_name,
            )

            result = travel_app.invoke(state_input, config=config)

        if result.get("user_name"):
            st.session_state.user_name = result["user_name"]

        answer = result["answer"]
        st.write(answer)

        sources = result.get("sources", [])
        if sources:
            with st.expander("Sources consulted"):
                for src in sources:
                    st.markdown(f"• **{src}**")

        with st.expander("Debug info"):
            st.markdown(f"**Route taken:** `{result.get('route', 'N/A')}`")
            faith = result.get("faithfulness", 1.0)
            color = "green" if faith >= 0.8 else "orange" if faith >= 0.6 else "red"
            st.markdown(f"**Faithfulness score:** :{color}[{faith:.2f}]")
            st.markdown(f"**Eval retries:** `{result.get('eval_retries', 0)}`")
            if result.get("tool_result"):
                st.markdown(f"**Tool output:** {result['tool_result'][:200]}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "faithfulness": result.get("faithfulness"),
    })
    st.rerun()
