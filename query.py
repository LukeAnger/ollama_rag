import argparse
from langchain.vectorstores.chrome import Chroma
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