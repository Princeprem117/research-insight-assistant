'''from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()


def create_llm():

    llm = ChatOpenRouter(
        model="apodex/apodex-1.1-mini:free",
        temperature=0,
    )

    return llm

'''

import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


load_dotenv()


def create_llm():

    api_key = os.getenv("OPENROUTER_APIKEY")

    print("Key loaded:", bool(api_key))
    print("Key prefix:", api_key[:10] if api_key else None)

    llm = ChatOpenRouter(
        model="apodex/apodex-1.1-mini:free",
        api_key=api_key,
        temperature=0,
    )

    return llm