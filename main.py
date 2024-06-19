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
    parser = argparse.ArgumentParser(description='Create a vector database from a directory of PDFs')   
    parser.add_argument('--reset', action='store_true', help='Reset the vector database')
    parser.add_argument('pdf_dir', help='Directory of PDFs')
    args = parser.parse_args()

    if args.reset:
        print('Resetting vector database')
        reset_vector_db()

    documents = load_documents(args.pdf_dir)
    chunks = split_documents(documents)
    add_documents_to_vector_db(chunks)


def load_documents(pdf_dir):
    document_loader = PyPDFDirectoryLoader(pdf_dir)
    documents = document_loader.load()

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    documents = splitter.split_documents(documents)

    return documents


def add_documents_to_vector_db(chunks):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())

    for chunk in chunks:
        db.add_document(chunk)
    db.persist()


def reset_vector_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # os.makedirs(CHROMA_PATH) 

if __name__ == '__main__':
    main()