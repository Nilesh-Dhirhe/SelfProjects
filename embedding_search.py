import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class EmbeddingSearch:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.embeddings = self.vectorizer.fit_transform(documents).toarray()
        self.embeddings = normalize(self.embeddings)

    def search(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        query_vec = query_vec / np.linalg.norm(query_vec)
        scores = [cosine_similarity(query_vec, emb) for emb in self.embeddings]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(i, scores[i]) for i in top_indices]

if __name__ == "__main__":
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is amazing for pattern recognition",
        "Natural language processing deals with text data"
    ]

    search_engine = EmbeddingSearch(docs)
    results = search_engine.search("text analysis and processing", top_k=2)

    for idx, score in results:
        print(f"Doc {idx} - Score: {score:.4f} - {docs[idx]}")
