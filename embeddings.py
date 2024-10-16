from langchain.embeddings  import HuggingFaceEmbeddings

def get_embeddings():
    embeddings = HuggingFaceEmbeddings()

    return embeddings
