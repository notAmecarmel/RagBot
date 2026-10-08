import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

question = input("Question: ")

results = vectorstore.similarity_search(
    question,
    k=4
)

context = "\n\n".join(
    doc.page_content
    for doc in results
)

prompt = f"""
You are a question-answering assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)