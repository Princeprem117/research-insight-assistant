import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


load_dotenv()


def create_llm(
    model: str | None = None,
    temperature: float = 0,
) -> ChatOpenRouter:
    """Create an LLM instance.

    Args:
        model: Model name (defaults to OPENROUTER_MODEL env var or nvidia/nemotron-3.5-lightning:free).
        temperature: Sampling temperature.

    Returns:
        ChatOpenRouter instance.
    """
    api_key = os.getenv("OPENROUTER_APIKEY")

    if not api_key:
        raise ValueError("OPENROUTER_APIKEY is not set in .env")

    # Default model: use env var or fallback to free model
    if model is None:
        model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")

    llm = ChatOpenRouter(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )

    return llm