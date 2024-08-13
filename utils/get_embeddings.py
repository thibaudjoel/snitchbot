from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    model_name = "intfloat/multilingual-e5-large"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings
