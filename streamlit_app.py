
import uuid

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Research Insight Assistant",
    page_icon="📚",
    layout="wide",
)


# Initialize conversation state.
if "conversations" not in st.session_state:
    first_id = str(uuid.uuid4())

    st.session_state.conversations = {
        first_id: {
            "title": "New conversation",
            "messages": [],
        }
    }
    st.session_state.active_conversation = first_id


conversations = st.session_state.conversations


def create_conversation():
    conversation_id = str(uuid.uuid4())

    conversations[conversation_id] = {
        "title": "New conversation",
        "messages": [],
    }

    st.session_state.active_conversation = conversation_id


# Sidebar: conversation management.
with st.sidebar:
    st.title("📚 Research Assistant")

    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):
        create_conversation()
        st.rerun()

    st.divider()
    st.subheader("Your conversations")

    # Show newest conversations first.
    for conversation_id, conversation in reversed(
        list(conversations.items())
    ):
        is_active = (
            conversation_id
            == st.session_state.active_conversation
        )

        if st.button(
            ("● " if is_active else "○ ")
            + conversation["title"],
            key=f"conversation_{conversation_id}",
            use_container_width=True,
        ):
            st.session_state.active_conversation = (
                conversation_id
            )
            st.rerun()

    st.divider()
    st.caption("Powered by LangChain, FastAPI and Chroma")


# Get the selected conversation.
active_id = st.session_state.active_conversation
active_conversation = conversations[active_id]
messages = active_conversation["messages"]


# Main interface.
st.title("Research Insight Assistant")
st.caption(
    "Explore your research documents using conversational RAG."
)


# Display the selected conversation's messages.
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.write(
                        f"**{source['file']}** "
                        f"— Page {source['page']}"
                    )


question = st.chat_input(
    "Ask a question about your research PDFs..."
)


if question:
    # Set the title using the first question.
    if not messages:
        active_conversation["title"] = question[:40]
        if len(question) > 40:
            active_conversation["title"] += "..."

    messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching documents and generating an answer..."
            ):
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

            answer = result["answer"]
            sources = result.get("sources", [])

            st.markdown(answer)

            if sources:
                with st.expander("View sources"):
                    for source in sources:
                        st.write(
                            f"**{source['file']}** "
                            f"— Page {source['page']}"
                        )

            messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Start it using: "
                "`python -m uvicorn api:app --reload`"
            )

        except requests.exceptions.Timeout:
            st.error(
                "The request timed out. "
                "Check the FastAPI terminal and try again."
            )

        except requests.exceptions.HTTPError as exc:
            st.error(f"API request failed: {exc}")

        except (ValueError, KeyError) as exc:
            st.error(f"Unexpected API response: {exc}")
