from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

load_dotenv()

# 1. Load the same embedding model used during ingestion
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 2. Load existing Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# 4. Get user's question
question = input("\nAsk a question about the document: ")

# 5. Retrieve relevant chunks
results = vectorstore.similarity_search(
    question,
    k=4
)

# 6. Combine retrieved chunks
context = "\n\n".join(
    doc.page_content
    for doc in results
)

# 7. Build RAG prompt
prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the context provided below.

If the answer cannot be found in the context,
say: "The information is not available in the document."

Context:
{context}

Question:
{question}

Answer:
"""

# 8. Generate answer
response = llm.invoke(prompt)

print("\n--- ANSWER ---")
print(response.content)