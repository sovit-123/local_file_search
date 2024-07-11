"""
Search based on cosine similarity.
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
        print(f"Filename: {result[0]['filename']}, Score: {result[1]}")
        # print(f"Content: {result[0]['content'][:500]}...\n")

if __name__ == "__main__":
    main()
