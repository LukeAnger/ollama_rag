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

Answer the question based on the above context {question} and provide a detailed explanation.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', type=str, help='The question')
    args = parser.parse_args()
    question = args.question
    query_rag(question)


def query_rag(question):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())

    all_docs = db.get()
    # print("All documents:", all_docs)
    model = Ollama(model='llama3')

    sim_search = db.similarity_search_with_score(question, k=10)
    print("Similarity search results:", sim_search)

    context = sim_search[0][0].page_content

    # context = ''
    # for i in range(0, len(sim_search)):
    #     context += sim_search[i][0].page_content + ' '

    print("---------------------------------------------------------------------------------------------------------------------------------------------------------")

    print("Context:", context)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)

    response = model.invoke(prompt)
    print("Response:", response)
    return response

if __name__ == '__main__':
    main()