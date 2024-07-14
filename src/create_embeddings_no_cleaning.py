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
import argparse

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--add-file-content',
    dest='add_file_content',
    action='store_true',
    help='whether to store the file content in the final index file or not'
)
parser.add_argument(
    '--index-file-name',
    dest='index_file_name',
    help='file name for the index JSON file',
    required=True
)
args = parser.parse_args()

# Load SBERT model
model_id = 'all-MiniLM-L6-v2'
# model_id = 'outputs/checkpoint-12500' # Or any other custom model path.
model = SentenceTransformer(model_id)
# Device setup (not needed for SentenceTransformer as it handles it internally)
device = model.device
print(device)

# -1 = embed all files
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

def chunk_text(text, chunk_size=512, overlap=50):
    """Chunk the text into overlapping windows."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def encode_document(
    filename, documents, add_file_content, content, chunk_size=512, overlap=50
):
    """Encode the document in chunks."""
    chunks = chunk_text(content, chunk_size, overlap)

    if not chunks:  # If no chunks are possible.
        features = extract_features(content).tolist()
        if add_file_content: # If original file content to be added.
            documents.append({
                'filename': filename, 
                'chunk': 0, 
                'content': content, 
                'features': features
            })
        else:
            documents.append({
                'filename': filename, 
                'chunk': 0, 
                'features': features
            })

    else:
        for i, chunk in enumerate(chunks):
            features = extract_features(chunk).tolist()
            if add_file_content: # If original file content to be added.
                documents.append({
                    'filename': filename, 
                    'chunk': i, 
                    'content': chunk, 
                    'features': features
                })
            else:
                documents.append({
                    'filename': filename, 
                    'chunk': i, 
                    'features': features
                })

    return documents

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

                documents = encode_document(
                    filename, 
                    documents, 
                    args.add_file_content, 
                    content, 
                    chunk_size=512,
                    overlap=50 
                )
                
    return documents

# Example usage
documents = load_and_preprocess_text_files('../data/paper_files')

# Save documents with embeddings to a JSON file
with open(os.path.join('..', 'data', args.index_file_name), 'w') as f:
    json.dump(documents, f)