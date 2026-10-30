import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Research Insight Assistant",
    page_icon="📚",
    layout="wide",
)


def api_get(path):
    response = requests.get(
        f"{API_URL}{path}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def create_conversation():
    response = requests.post(
        f"{API_URL}/conversations",
        json={"title": "New conversation"},
        timeout=30,
    )
    response.raise_for_status()
    conversation = response.json()
    st.session_state.active_conversation = conversation["id"]


# Load saved conversations from SQLite through FastAPI.
try:
    conversations = api_get("/conversations")
except requests.RequestException:
    st.error(
        "Cannot connect to FastAPI. Start it with "
        "`python -m uvicorn api:app --reload`."
    )
    st.stop()


if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None

if (
    st.session_state.active_conversation
    and not any(
        c["id"] == st.session_state.active_conversation
        for c in conversations
    )
):
    st.session_state.active_conversation = None

if not conversations:
    try:
        create_conversation()
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Could not create conversation: {exc}")
        st.stop()

if st.session_state.active_conversation is None:
    st.session_state.active_conversation = conversations[0]["id"]


# Sidebar.
with st.sidebar:
    st.title("📚 Research Assistant")

    if st.button("＋ New conversation", use_container_width=True):
        try:
            create_conversation()
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not create conversation: {exc}")

    st.divider()
    st.subheader("Your conversations")

    for conversation in conversations:
        conversation_id = conversation["id"]
        title = conversation.get("title") or "New conversation"
        active = (
            conversation_id == st.session_state.active_conversation
        )

        if st.button(
            ("● " if active else "○ ") + title,
            key=f"conversation_{conversation_id}",
            use_container_width=True,
        ):
            st.session_state.active_conversation = conversation_id
            st.rerun()

    st.divider()
    st.caption("Powered by LangChain, FastAPI and Chroma")


active_id = st.session_state.active_conversation

st.title("Research Insight Assistant")
st.caption("Explore your research documents using conversational RAG.")


# Load messages for the selected conversation.
try:
    messages = api_get(
        f"/conversations/{active_id}/messages"
    )
except requests.RequestException as exc:
    st.error(f"Could not load conversation: {exc}")
    st.stop()


for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.write(
                        f"**{source['file']}** — Page {source['page']}"
                    )


question = st.chat_input("Ask a question about your research PDFs...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "session_id": active_id,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()

            st.markdown(result["answer"])

            sources = result.get("sources", [])
            if sources:
                with st.expander("View sources"):
                    for source in sources:
                        st.write(
                            f"**{source['file']}** — Page {source['page']}"
                        )

            # Reload from SQLite so the sidebar title and history update.
            st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI.")
        except requests.exceptions.Timeout:
            st.error("The request timed out. Check the backend.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"API request failed: {exc}")
