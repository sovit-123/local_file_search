"""
Script to create SBERT embedding and store in JSON file. 
We clean the text using a SpaCy model here.

NOTE:
For joblib multiprocessing, processes need to run on CPU.
However, using joblib + multiprocessing + cpu is faster
compared to non-parallel processing on GPU.
For examples, on a laptop RTX 3070Ti, non-parallel embedding 
creation of 7000 research paper text files takes around 60 minutes.
On CPU (for both spacy and sentence transformer models) with joblib n_jobs=16
the same operation takes around 8 minutes.  
Spoiler: The majority of the time is consumed by clean up of the documents
done using Spacy.

Requirements:
$ pip install spacy
$ python -m spacy download en_core_web_sm

USAGE:
$ python create_embeddings.py
"""

import os
import spacy
import torch
import json

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from joblib import Parallel, delayed

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

# -1 = embed all files
total_files_to_embed = -1

# Load SpaCy model
if device == 'cuda' or device == 'cuda:0':
    spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')

# To store the document embeddings.
documents = []

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

def load_and_preprocess_text_files(directory, filename):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename), 'r', errors='ignore') as file:
            content = file.read()
            preprocessed_content = preprocess_text(content)
            features = extract_features(preprocessed_content).tolist()
            # documents.append({'filename': filename, 'content': content, 'preprocessed_content': preprocessed_content})
            return {'filename': filename, 'features': features}
    return None

# Load SBERT model
model_id = 'all-MiniLM-L6-v2'
# model_id = 'outputs/checkpoint-12500' # Or any other custom model path.
model = SentenceTransformer(model_id).to(device)
print(f"SBERT model device: {next(model.parameters()).device}")

directory = '../data/paper_files'
all_files = os.listdir(directory)
all_files.sort()
if total_files_to_embed > -1:
    files_to_embed = all_files[:total_files_to_embed]
else: 
    files_to_embed = all_files

results = Parallel(n_jobs=16, backend='multiprocessing')(
    delayed(load_and_preprocess_text_files)(directory, filename) \
        for filename in tqdm(files_to_embed, total=len(files_to_embed))
)

# Filter out None values
documents = [result for result in results if result is not None]

# Save documents with embeddings to a JSON file
with open('../data/cleaned_indexed_documents_multiprocessing_pretrained.json', 'w') as f:
    json.dump(documents, f)