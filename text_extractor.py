import requests
from bs4 import BeautifulSoup
import logging
from transformers.utils import logging as hf_logging
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# --- Scrape webpage and save article text ---
def scrape_webpage():
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"  # hard-coded URL
    print(f"Attempting to fetch URL: {url}")

    try:
        response = requests.get(url)
        print(f"HTTP response status code: {response.status_code}")

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find('div', class_='mw-parser-output')
            if not content_div:
                print("⚠️ Warning: Could not find <div class='mw-parser-output'>")
                return ""

            paragraphs = content_div.find_all('p')
            print(f"Found {len(paragraphs)} <p> tags")

            article_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

            with open("Selected_Document.txt", "w", encoding="utf-8") as f:
                f.write(article_text)

            print(f"✅ Successfully saved content to Selected_Document.txt")
            return article_text
        else:
            print(f"❌ Failed to fetch webpage. Status code: {response.status_code}")
            return ""
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return ""

def main():
    print("Starting scrape_webpage()...")
    scrape_webpage()
    print("scrape_webpage() finished.")

# --- Logging and warnings configuration ---
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# --- Variables ---
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 5

# --- Read saved document ---
with open("Selected_Document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --- Text splitting ---
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' ', ''],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = text_splitter.split_text(text)

# --- Set up generator pipeline ---
generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)

# --- Placeholder FAISS index setup ---
# (You need to implement FAISS index creation and embedding for real use)
# Here we simulate chunks with indices for retrieval demonstration
faiss_index = None  # Placeholder; implement FAISS index with embeddings

def retrieve_chunks(question, k=top_k):
    # TODO: encode the question using the embedding model,
    # search the FAISS index, return top k chunks
    # Placeholder: just return first k chunks for demo
    return chunks[:k]

def answer_question(question):
    relevant_chunks = retrieve_chunks(question, k=top_k)
    context = "\n\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    generated = generator(prompt, max_length=200, do_sample=False)
    return generated[0]['generated_text']

if __name__ == '__main__':
    main()

    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break
        print("Answer:", answer_question(question))
