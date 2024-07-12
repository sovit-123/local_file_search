"""
Search based on cosine similarity.

USAGE:
$ python search_cosine.py --index-file <path/to/index.json file>
"""

import json
import argparse

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--index-file',
    dest='index_file',
    required=True,
    help='path to an indexed embedding JSON file'
)
args = parser.parse_args()

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('outputs/checkpoint-12500')
print(model)

def extract_features(text):
    """Generate SBERT embeddings for the input text."""
    return model.encode(text)

def process_query(query):
    """Preprocess the query and generate SBERT embeddings."""
    query_features = extract_features(query).tolist()
    return query_features

def search(query, documents, top_k=5):
    """Search for the most relevant documents to the query."""
    query_features = process_query(query)
    scores = []
    for document in documents:
        score = cosine_similarity([query_features], [document['features']])[0][0]
        scores.append((document, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def chunk_text(text, chunk_size=100, overlap=50):
    """Chunk the text into overlapping windows."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def extract_relevant_part(query, content, chunk_size=256, overlap=50):
    """Extract the part of the content that is most relevant to the query."""
    chunks = chunk_text(content, chunk_size, overlap)
    if not chunks:
        return content  # Return full content if it can't be split

    chunk_embeddings = model.encode(chunks)
    query_embedding = extract_features(query)
    scores = cosine_similarity([query_embedding], chunk_embeddings).flatten()
    best_chunk_idx = scores.argmax()
    return chunks[best_chunk_idx]


def load_documents(file_path):
    """Load preprocessed documents and embeddings from a JSON file."""
    with open(file_path, 'r') as f:
        documents = json.load(f)
    return documents

def main():
    # File paths
    # documents_file_path = '../data/indexed_documents_finetuned.json'
    # documents_file_path = '../data/indexed_documents_pretrained.json'
    documents_file_path = args.index_file

    # Load documents.
    documents = load_documents(documents_file_path)

    # Example query
    query = input("Enter your search query: ")

    # Perform search
    results = search(query, documents)
    for result in results:
        document = result[0]
        print(f"Filename: {result[0]['filename']}, Score: {result[1]}")
        relevant_part = extract_relevant_part(query, document['content'])
        print(f"Relevant part: {relevant_part}\n")

if __name__ == "__main__":
    main()
