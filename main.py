import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embeddings import get_embeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH='chroma'
DATA_PATH='../data'

def main():
    print("Starting main")
    parser = argparse.ArgumentParser(description='Create a vector database from a directory of PDFs')   
    parser.add_argument('--reset', action='store_true', help='Reset the vector database')
    parser.add_argument('pdf_dir', help='Directory of PDFs')
    args = parser.parse_args()
    print(args)
    if args.reset:
        print('Resetting vector database')
        reset_vector_db()

    documents = load_documents(args.pdf_dir)
    chunks = split_documents(documents)
    add_documents_to_vector_db(chunks)


def load_documents(pdf_dir):
    print("Loading documents")
    document_loader = PyPDFDirectoryLoader(pdf_dir)
    documents = document_loader.load()

    return documents


def split_documents(documents):
    print("Splitting documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    documents = splitter.split_documents(documents)

    return documents


def add_documents_to_vector_db(chunks):
    print("Adding documents to vector database")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    chunks = chunks[:50]
    chunks = assign_chunk_ids(chunks)

    #batch add documents
    for i in range(0, len(chunks), 10):
        db.add_documents(chunks[i:i+100], ids=[c.metadata['id'] for c in chunks[i:i+100]])
        # print("Added documents:", db.get())

    all_docs = db.get().documents
    print("Added documents:", all_docs[0])
    print("Added documents final:", len(all_docs))

def assign_chunk_ids(chunks):
    for i, chunk in enumerate(chunks):
        source = chunk.metadata['source']
        page = chunk.metadata['page']
        chunk.metadata['id'] = f'{source}_{page}_{i}'
        print("chunk: ", chunk.metadata['id'])

    return chunks



def reset_vector_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # os.makedirs(CHROMA_PATH) 

if __name__ == '__main__':
    main()