from langchain_core.prompts import ChatPromptTemplate

def create_rag_prompt():

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a research assistant.

            Answer the user's question using only the provided context.

            If the answer cannot be found in the context, say:
            "I could not find the answer in the provided research document."

            Context:
            {context}
            """
        ),
        (
            "human",
            "{question}"
        )
    ])

    return prompt