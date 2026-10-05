from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


def create_llm():

    llm = ChatOpenAI(
        model="apodex/apodex-1.1-mini:free",
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

    return llm