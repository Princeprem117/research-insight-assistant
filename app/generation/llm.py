import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


load_dotenv()


def create_llm():

    api_key = os.getenv("OPENROUTER_APIKEY")

    print("Key loaded:", bool(api_key))
    print("Key prefix:", api_key[:10] if api_key else None)

    if not api_key:
        raise ValueError("OPENROUTER_APIKEY is not set in .env")

    llm = ChatOpenRouter(
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
        temperature=0,
    )

    return llm