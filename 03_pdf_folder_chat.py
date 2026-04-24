#!pip install pdfplumber langchain langchain-community chromadb langchain_text_splitters sentence-transformers ollama

model_name = 'medgemma:4b'

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Imports
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama
import os


rebuild_database = False
folder_path = "./pdfs"   # <-- change to your folder path



def load_pdfs_from_folder(folder_path: str) -> list[tuple[str, str]]:
    """Returns a list of (text, source_filename) tuples for all PDFs in folder."""
    results = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {folder_path}")

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        with pdfplumber.open(path) as pdf:
            text = "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
        results.append((text, filename))
        print(f"  Loaded '{filename}': {len(text):,} characters")

    return results

# --- Load all PDFs ---
if rebuild_database:
    PDF_FOLDER = folder_path   # <-- change to your folder path
    print(f"Scanning for PDFs in: {PDF_FOLDER}")
    pdf_docs = load_pdfs_from_folder(PDF_FOLDER)
    print(f"Loaded {len(pdf_docs)} PDF(s)\n")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

if rebuild_database:
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
    all_chunks = []
    all_metadatas = []

    for text, filename in pdf_docs:
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        all_metadatas.extend([{"source": filename}] * len(chunks))
        print(f"  '{filename}': {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # --- Build vectorstore ---

    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=f"{folder_path}/chroma_db"  # Save vectostore to disk for future use
    )   # each chunk tagged with its source PDF
else:
    # Load existing vectorstore from Chroma persistence
    try:
        vectorstore = Chroma(
            persist_directory=f"{folder_path}/chroma_db",
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        print(f"Vector store loaded from '{folder_path}/chroma_db'")
    except Exception as e:
        print(f"No existing vector store found. Error: {e}")
        print("Please set rebuild_database=True to create it.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print("Vector store ready ✓")


# Chat function
def chat(question: str, history: list[dict]) -> str:
    # Retrieve relevant chunks from the PDF
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Build the prompt
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the context below. If the answer is not in the context, say so.\n\n"
        f"CONTEXT:\n{context}"
    )

    # Append this turn to history
    history.append({"role": "user", "content": question})

    response = ollama.chat(
        model=model_name,   # or "gemma:2b", "gemma:7b", etc.
        messages=[{"role": "system", "content": system_prompt}] + history,
        options={"num_ctx": 32768} 
    )

    answer = response["message"]["content"]
    history.append({"role": "assistant", "content": answer})
    return answer

# Interactive chat loop (run this cell to start chatting)
history = []

# Clear console
os.system('clear')

print("PDF Chatbot ready. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue
    answer = chat(question, history)
    print(f"\nLLM: {answer}\n")
