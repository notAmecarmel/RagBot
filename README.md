# 📄 RAG PDF Question Answering System

A Retrieval-Augmented Generation (RAG) system that allows users to ask questions about a PDF document and receive answers grounded in the document's content.

The system retrieves the most relevant sections of the document before sending them to the LLM, rather than providing the entire document as context.

## 🚀 Features

- PDF document ingestion
- Automatic document chunking
- Gemini-powered text embeddings
- ChromaDB vector storage
- Semantic similarity search
- Top-k relevant chunk retrieval
- Gemini-powered answer generation
- Context-grounded responses

## 🏗️ Architecture

```text
PDF Document
     ↓
PyPDFLoader
     ↓
Document Extraction
     ↓
Recursive Text Splitting
     ↓
Document Chunks
     ↓
Gemini Embeddings
     ↓
ChromaDB
     ↓
User Query
     ↓
Query Embedding
     ↓
Similarity Search
     ↓
Top-K Relevant Chunks
     ↓
RAG Prompt
     ↓
Gemini LLM
     ↓
Final Answer
```

## 🧠 How It Works

### 1. Document Ingestion

The PDF is loaded using `PyPDFLoader` and converted into LangChain `Document` objects containing the document text and metadata such as page number and source.

### 2. Chunking

The extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

```text
Chunk Size: 1000
Chunk Overlap: 200
```

Chunking allows the retrieval system to search smaller, more relevant sections instead of processing the entire document.

### 3. Embeddings

Each document chunk is converted into a numerical vector using Google's `gemini-embedding-001` embedding model.

These vectors represent the semantic meaning of the text and allow the system to perform semantic search.

### 4. Vector Database

The generated embeddings are stored in ChromaDB along with the original text and metadata.

ChromaDB is used to efficiently search for chunks that are semantically similar to a user's query.

### 5. Retrieval

When a user asks a question:

1. The question is converted into an embedding.
2. ChromaDB performs similarity search against the stored document embeddings.
3. The most relevant chunks are retrieved.
4. The top-k chunks are provided as context to the LLM.

### 6. Generation

The retrieved chunks are inserted into a prompt along with the user's question.

Gemini then generates an answer based on the retrieved document context.

This creates the complete:

```text
Retrieve → Augment → Generate
```

RAG pipeline.

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **Google Gemini**
- **Gemini Embeddings**
- **ChromaDB**
- **PyPDF**
- **python-dotenv**

## 📁 Project Structure

```text
rag-project/
│
├── data/
│   └── document.pdf
│
├── chroma_db/
│   └── Vector database files
│
├── ingest.py
├── rag.py
├── .env
└── README.md
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag-project
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file:

```text
GOOGLE_API_KEY=your_api_key_here
```

Do not commit the `.env` file to GitHub.

Add it to `.gitignore`:

```text
.env
.venv/
chroma_db/
__pycache__/
```

### 5. Add a PDF

Place your PDF inside:

```text
data/
```

Update the PDF path in `ingest.py` if necessary.

### 6. Create the vector database

Run:

```bash
python ingest.py
```

This loads the PDF, chunks the text, generates embeddings and stores them in ChromaDB.

### 7. Run the RAG system

```bash
python rag.py
```

Enter a question about the document when prompted.

## 💡 Example

```text
Ask a question about the document: What is the main objective of the research?

--- ANSWER ---

The main objective of the research is ...
```

## 🔍 Why RAG?

Traditional LLMs may not have access to private or domain-specific documents.

RAG solves this by retrieving relevant information from an external knowledge source and providing it to the LLM at inference time.

Advantages include:

- Reduced reliance on the model's internal knowledge
- Better grounding in source documents
- Ability to work with private/custom information
- No model retraining required
- More efficient than sending an entire large document to the LLM

## 🆚 RAG vs Fine-Tuning

| RAG | Fine-Tuning |
|---|---|
| Retrieves external information | Trains model on additional examples |
| Knowledge can be updated easily | Updating knowledge requires retraining |
| Good for document-based Q&A | Good for changing model behavior/style |
| Does not modify model weights | Modifies model weights |

## 🔮 Future Improvements

- Metadata-based filtering
- Hybrid keyword + semantic search
- Reranking retrieved documents
- Improved chunking strategies
- Retrieval and answer evaluation
- Source/page citations in responses
- Conversational memory
- Support for multiple documents
- OCR support for scanned PDFs

## 📌 Key Concepts Demonstrated

- Document ingestion
- Text chunking
- Embeddings
- Vector databases
- Semantic search
- Similarity retrieval
- Top-k retrieval
- Prompt augmentation
- Grounded generation
- Retrieval-Augmented Generation (RAG)

## 👨‍💻 Author

**Ameya Ingale**

Built as a practical implementation to understand and demonstrate the architecture and core concepts behind Retrieval-Augmented Generation systems.