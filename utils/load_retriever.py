from langchain_chroma import Chroma
from langchain_core.runnables import chain

from .get_embeddings import get_embeddings


def load_retriever():

    embedding_function = get_embeddings()

    vector_store = Chroma(
        persist_directory="./chroma_langchain_db",
        collection_name="abstimmungsergebnisse",
        embedding_function=embedding_function,
    )

    @chain
    def retriever(query: str):
        try:
            docs, scores = zip(*vector_store.similarity_search_with_score(query))
            docms = [docs[i] for i in range(len(docs)) if scores[i] < 0.3]
        except:
            return []

        return docms

    return retriever
