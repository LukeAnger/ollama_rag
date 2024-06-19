import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from embeddings import get_embeddings

CHROMA_PATH='chroma'

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', type=str, help='The question')
    args = parser.parse_args()
    question = args.question

def query_rag(question, context):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    model = Ollama(model='llama:7b')

    prompt = ChatPromptTemplate(PROMPT_TEMPLATE)
    prompt = prompt.render(context=context, question=question)

    response = model.invoke(prompt)

    return response