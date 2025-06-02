import logging
import warnings
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Logging and warnings configuration ---
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# --- RAG Setup ---
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 5

# Read in pre-extracted text
with open("Selected_Document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunk the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_text(text)

# Generate embeddings for each chunk
embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load generator model
generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

def retrieve_chunks(question, k=top_k):
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0]]

def answer_question(question):
    context = "\n\n".join(retrieve_chunks(question))
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = generator(prompt, max_length=200, do_sample=False)
    return result[0]["generated_text"]

if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))
