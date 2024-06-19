from langchain_community.embeddings import OllamaEmbeddings

def get_embeddings():
    embeddings = OllamaEmbeddings(model='llama:7b')

    return embeddings
