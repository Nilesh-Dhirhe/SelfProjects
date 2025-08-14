import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK data if not available
nltk.download('punkt')
nltk.download('stopwords')

# Custom stopwords
CUSTOM_STOPWORDS = set(stopwords.words('english')) | {"example", "etc"}

def clean_text(text):
    """Lowercase, remove special chars, numbers."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove digits
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_filter(text):
    """Tokenize and remove stopwords."""
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in CUSTOM_STOPWORDS]
    return filtered

def chunk_tokens(tokens, chunk_size=50):
    """Chunk tokens into fixed size pieces."""
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

if __name__ == "__main__":
    sample_text = """This is an example sentence! It contains numbers like 123, and punctuation marks."""
    cleaned = clean_text(sample_text)
    tokens = tokenize_and_filter(cleaned)
    chunks = chunk_tokens(tokens, chunk_size=5)

    print("Original:", sample_text)
    print("Cleaned:", cleaned)
    print("Tokens:", tokens)
    print("Chunks:", chunks)
