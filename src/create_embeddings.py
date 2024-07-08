"""
Script to create SBERT embedding and store in JSON file. 
We clean the text using a SpaCy model here.

Requirements:
pip install spacy
python -m spacy download en_core_web_sm
"""

import os
import spacy
import torch
import json

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# -1 = all files
total_files_to_embed = -1

spacy.prefer_gpu()
# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.text for token in doc if \
        # not token.is_stop \
        # and not token.is_punct \
        not token.is_space \
        and not token.is_digit
    ]
    return ' '.join(tokens)

def extract_features(text):
    """
    Extracts embeddings from a given text using the SentenceTransformer model.
    
    :param text: The text to embed.

    Returns:
        embeddings: A list of embeddings.
    """
    embeddings = model.encode(text)
    return embeddings

def load_and_preprocess_text_files(directory):
    documents = []
    all_files = os.listdir(directory)
    all_files.sort()
    if total_files_to_embed > -1:
        files_to_embed = all_files[:total_files_to_embed]
    else: 
        files_to_embed = all_files

    # print(files_to_embed)
    for filename in tqdm(files_to_embed, total=len(os.listdir(directory))):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', errors='ignore') as file:
                content = file.read()
                preprocessed_content = preprocess_text(content)
                features = extract_features(preprocessed_content).tolist()
                # documents.append({'filename': filename, 'content': content, 'preprocessed_content': preprocessed_content})
                documents.append({'filename': filename, 'features': features})
    return documents

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
print(f"SBERT model device: {next(model.parameters()).device}")

documents = load_and_preprocess_text_files('../data/paper_files')

# Save documents with embeddings to a JSON file
with open('../data/cleaned_indexed_documents.json', 'w') as f:
    json.dump(documents, f)