from langchain_community.vectorstores import Chroma

def get_all_docs():
    db = Chroma(persist_directory='chroma')
    query = db.get()
    return query

# print("DICT KEYS: ", get_all_docs().keys())
print("DOCUMENTS: ", get_all_docs()['documents'][:5])