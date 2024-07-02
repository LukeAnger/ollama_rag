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

## Implement some way to have a persistent chatbot that can answer questions based on a context and a question

def reset_vector_db():
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    db.reset()

def main():
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    model = Ollama(model='llama3')
    
    while True:
        question = input("Enter your question: ")
        sim_search = db.similarity_search_with_score(question, k=25)
        # print("Similarity search results:", sim_search)

        # context = sim_search[0][0].page_content
        # concat first 5 of sim search
        context = ''
        for i in range(0, 5):
            context += sim_search[i][0].page_content + ' '
        

        # print("---------------------------------------------------------------------------------------------------------------------------------------------------------")

        # print("Context:", context)

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=question)

        response = model.invoke(prompt)
        print("Response:", response)
        return response

if __name__ == '__main__':
    main()