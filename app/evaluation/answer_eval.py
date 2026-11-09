import csv
import json
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.evaluation.test_questions import TEST_QUESTIONS
from app.ingestion.embedder import create_embedding_model
from app.vectorstore.store import create_vector_store
from app.retrieval.retriever import create_retriever
from app.generation.llm import create_llm
from app.generation.prompt import (
    create_rag_prompt,
    create_history_aware_prompt,
)
from app.rag.chain import create_rag_chain, format_docs


OUTPUT_DIR = Path("evaluation_results")
OUTPUT_FILE = OUTPUT_DIR / "answer_quality.csv"


# LLM-as-a-judge evaluation prompt.
judge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an evaluator for a Retrieval-Augmented Generation (RAG) system.

Evaluate the generated answer using the question and retrieved context.

Evaluate these two dimensions independently:

1. Faithfulness:
   - Is every factual claim in the answer supported by the retrieved context?
   - A score of 5 means fully supported.
   - A score of 1 means mostly unsupported or contradictory.
   - Do not use outside knowledge to justify the answer.

2. Answer relevance:
   - Does the answer directly address the user's question?
   - A score of 5 means highly relevant and directly answers the question.
   - A score of 1 means irrelevant or does not answer the question.

Use integer scores from 1 to 5.

Return ONLY a valid JSON object with this exact structure:

{{
    "faithfulness_score": 1,
    "faithfulness_reason": "Explanation",
    "relevance_score": 1,
    "relevance_reason": "Explanation"
}}
"""
    ),
    (
        "human",
        """
Question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}
"""
    ),
])


def validate_result(result):
    """Validate the judge's scores and explanations."""

    score_fields = [
        "faithfulness_score",
        "relevance_score",
    ]

    for field in score_fields:
        score = result.get(field)

        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 1 <= score <= 5
        ):
            raise ValueError(
                f"Invalid {field}: {score}. Expected an integer from 1 to 5."
            )

    for field in [
        "faithfulness_reason",
        "relevance_reason",
    ]:
        if not isinstance(result.get(field), str):
            raise ValueError(f"Invalid or missing field: {field}")

    return result


def evaluate_answers():

    # 1. Load the existing embedding model and Chroma collection.
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(embedding_model)

    collection = vector_store.get(include=["metadatas"])

    if not collection.get("ids"):
        raise ValueError(
            "Chroma index is empty. Index your PDFs before evaluation."
        )

    # 2. Create the same RAG chain used by the application.
    retriever = create_retriever(vector_store)
    llm = create_llm()

    rag_chain = create_rag_chain(
        retriever=retriever,
        prompt=create_rag_prompt(),
        llm=llm,
        history_aware_prompt=create_history_aware_prompt(),
    )

    # 3. Create the evaluation chain.
    evaluation_chain = (
        judge_prompt
        | llm
        | JsonOutputParser()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # 4. Generate and evaluate answers.
    for index, question in enumerate(TEST_QUESTIONS, start=1):

        print(f"\n[{index}/{len(TEST_QUESTIONS)}] {question}")

        # Empty history makes every test question independent.
        rag_result = rag_chain.invoke({
            "question": question,
            "chat_history": [],
        })

        answer = rag_result["answer"]
        documents = rag_result["documents"]
        context = format_docs(documents)

        print(f"Retrieved chunks: {len(documents)}")

        # Evaluate the generated answer.
        try:
            evaluation = evaluation_chain.invoke({
                "question": question,
                "context": context,
                "answer": answer,
            })

            evaluation = validate_result(evaluation)

        except Exception as e:
            print(f"[EVALUATION ERROR] {e}")

            evaluation = {
                "faithfulness_score": "",
                "faithfulness_reason": "Evaluation failed due to API error.",
                "relevance_score": "",
                "relevance_reason": "Evaluation failed due to API error.",
            }

        evaluation = validate_result(evaluation)

        results.append({
            "question": question,
            "answer": answer,
            "retrieved_chunks": len(documents),
            "faithfulness_score": evaluation["faithfulness_score"],
            "faithfulness_reason": evaluation["faithfulness_reason"],
            "relevance_score": evaluation["relevance_score"],
            "relevance_reason": evaluation["relevance_reason"],
        })

        print(
            "Faithfulness:",
            evaluation["faithfulness_score"],
            "-",
            evaluation["faithfulness_reason"],
        )

        print(
            "Answer relevance:",
            evaluation["relevance_score"],
            "-",
            evaluation["relevance_reason"],
        )

    # 5. Save detailed results.
def save_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question",
        "answer",
        "retrieved_chunks",
        "faithfulness_score",
        "faithfulness_reason",
        "relevance_score",
        "relevance_reason",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    results.append({
        # your existing result fields
    })

    save_results(results)

    # 6. Calculate average scores.
    average_faithfulness = sum(
        row["faithfulness_score"] for row in results
    ) / len(results)

    average_relevance = sum(
        row["relevance_score"] for row in results
    ) / len(results)

    print("\nANSWER QUALITY EVALUATION")
    print("-" * 45)

    print(f"Evaluated questions: {len(results)}")
    print(f"Average faithfulness: {average_faithfulness:.2f}/5")
    print(f"Average answer relevance: {average_relevance:.2f}/5")

    print(f"\nDetailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_answers()
