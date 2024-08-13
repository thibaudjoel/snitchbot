import getpass
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .get_llm import get_llm
from .get_prompt import get_prompt
from .load_retriever import load_retriever


def get_docs_from_question(question: str):
    load_dotenv()
    retriever = load_retriever()
    topic_from_question_prompt = get_prompt()
    llm = get_llm()
    rag_chain = (
        {"question": RunnablePassthrough()}
        | topic_from_question_prompt
        | llm
        | StrOutputParser()
        | retriever
    )

    retrieved_documents = rag_chain.invoke(question)

    return retrieved_documents


if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
    question = "Wie hat die SPD über Libanon abgestimmt?"
    get_docs_from_question(question)
