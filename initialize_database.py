from uuid import uuid4

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.get_embeddings import get_embeddings


def main():
    """
    Creates a vector store of the Abstimmungsthemen contained in the data.csv
    """
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="abstimmungsergebnisse",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    ).reset_collection()

    complete = pd.read_csv("data.csv")
    documents = (
        complete[["Abstimmungsthema", "Link"]]
        .drop_duplicates()
        .rename(columns={"Abstimmungsthema": "page_content", "Link": "metadata"})
        .to_dict("records")
    )

    for i, doc in enumerate(documents):
        documents[i] = Document(
            page_content=doc["page_content"],
            metadata={"source_link_xls": doc["metadata"]},
        )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)


if __name__ == "__main__":
    main()
