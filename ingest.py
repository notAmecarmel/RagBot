from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load PDF
loader = PyPDFLoader("data/greenbook.pdf")
documents = loader.load()

print(f"Number of pages: {len(documents)}")


# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# 3. Inspect a chunk
print("\n--- FIRST CHUNK ---")
print(chunks[0].page_content)

print("\n--- METADATA ---")
print(chunks[0].metadata)