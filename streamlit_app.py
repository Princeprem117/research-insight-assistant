
import uuid

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Research Insight Assistant",
    page_icon="📚",
    layout="wide",
)


# Initialize session state.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar.
with st.sidebar:
    st.title("📚 Research Assistant")
    st.write("Ask questions about your indexed research PDFs.")

    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by LangChain, FastAPI and Chroma")


# Main chat interface.
st.title("Research Insight Assistant")
st.caption("Explore your research documents using conversational RAG.")

# Display existing conversation.
for message in st.session_state.messages:
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
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
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
                            f"**{source['file']}** — Page {source['page']}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Start the backend with: "
                "`python -m uvicorn api:app --reload`"
            )

        except requests.exceptions.Timeout:
            st.error(
                "The request timed out. "
                "Try again or check the FastAPI terminal."
            )

        except requests.exceptions.HTTPError as exc:
            st.error(f"API request failed: {exc}")

        except (ValueError, KeyError) as exc:
            st.error(f"Unexpected API response: {exc}")
