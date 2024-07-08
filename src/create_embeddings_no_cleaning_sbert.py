"""
This script creates embeddings for text files in a directory and stores them in a JSON file.

**Usage:**

1. Specify the directory containing the text files in the `load_and_preprocess_text_files()` function.
2. Specify the filename for the JSON file in the `with open()` statement.

**Features:**

* Uses the SentenceTransformer model "all-MiniLM-L6-v2" to generate embeddings.
* Extracts features from each text file and stores them in a dictionary with keys "filename" and "features".
* Saves the dictionary of features to a JSON file.

**Notes:**

* The script assumes that all text files are in the same directory.
* The number of files to embed can be specified by setting the `total_files_to_embed` variable.
* This script does not handle errors in reading text files.
"""

import os
import json
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('outputs/checkpoint-12500')
# Device setup (not needed for SentenceTransformer as it handles it internally)
device = model.device
print(device)

# -1 = all files
total_files_to_embed = -1

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
    """
    Loads and preprocesses text files in a directory.

    :param directory: The directory containing the text files.

    Returns:
        documents: A list of dictionaries containing filename and embeddings.
    """
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
                features = extract_features(content).tolist()
                documents.append({'filename': filename, 'features': features})
    return documents

# Example usage
documents = load_and_preprocess_text_files('../data/paper_files')

# Save documents with embeddings to a JSON file
# with open('../data/indexed_documents_finetuned.json', 'w') as f:
#     json.dump(documents, f)

with open('../data/indexed_documents_pretrained.json', 'w') as f:
    json.dump(documents, f)