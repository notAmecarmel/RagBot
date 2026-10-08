from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("data/research-paper.pdf")
documents = loader.load()

print(f"Pages: {len(documents)}")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Chunks: {len(chunks)}")

# 3. Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 4. Create Chroma vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created successfully!")

# 5. Test retrieval
query = "What is the main objective of this research?"

results = vectorstore.similarity_search(
    query,
    k=3
)

print("\n--- RETRIEVED CHUNKS ---")

for i, result in enumerate(results):
    print(f"\nChunk {i + 1}:")
    print(result.page_content[:500])