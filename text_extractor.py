import requests  # For making HTTP requests
from bs4 import BeautifulSoup  # For parsing HTML
import logging
from transformers.utils import logging as hf_logging
import warnings

# --- Logging and warnings configuration ---
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def scrape_webpage():
    # URL of the Wikipedia article to scrape
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    print(f"Attempting to fetch URL: {url}")

    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)
        print(f"HTTP response status code: {response.status_code}")

        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the main content container in the Wikipedia page
            content_div = soup.find('div', class_='mw-parser-output')
            if not content_div:
                print("⚠️ Warning: Could not find <div class='mw-parser-output'>")
                return ""

            # Extract all paragraph tags within the main content
            paragraphs = content_div.find_all('p')
            print(f"Found {len(paragraphs)} <p> tags")

            # Clean and join paragraph text
            article_text = "\n\n".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

            # Save the cleaned text to a file
            with open("Selected_Document.txt", "w", encoding="utf-8") as f:
                f.write(article_text)

            print("✅ Successfully saved content to Selected_Document.txt")
            return article_text
        else:
            print(f"❌ Failed to fetch webpage. Status code: {response.status_code}")
            return ""
    except Exception as e:
        # Handle any errors during the request or processing
        print(f"❌ An error occurred: {e}")
        return ""


def main():
    print("Starting scrape_webpage()...")
    scrape_webpage()
    print("scrape_webpage() finished.")


if __name__ == '__main__':
    main()
