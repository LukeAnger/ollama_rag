from langchain_community.vectorstores import Chroma

def get_all_docs():
    db = Chroma(persist_directory='chroma')
    return db.get()

print(get_all_docs())