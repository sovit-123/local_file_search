"""
This script creates embeddings for text files in a directory and stores them 
in a JSON file. No cleaning of the text using Spacy. Recommened to use this 
script to generate embeddings since this gives better results when running
cosine similarity search on chunks of text.

USAGE:
* python create_embeddings_no_cleaning.py --index-file-name my_index_file.json
"""

import os
import json
import argparse

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from pypdf import PdfReader

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
parser.add_argument(
    '--directory-path',
    dest='directory_path',
    help='path to the directory either conteining text of PDF files',
    required=True
)
parser.add_argument(
    '--model',
    default='all-MiniLM-L6-v2',
    help='embedding model id from hugging face'
)
parser.add_argument(
    '--chunk-size',
    dest='chunk_size',
    default=512,
    type=int,
    help='chunk size of embedding creation and extracing content if needed'
)
parser.add_argument(
    '--overlap',
    default=50,
    type=int,
    help='text overlap when creating chunks'
)
args = parser.parse_args()

# Load SBERT model
model_id = args.model
# model_id = 'outputs/checkpoint-12500' # Or any other custom model path.
model = SentenceTransformer(model_id)
# Device setup (not needed for SentenceTransformer as it handles it internally)
device = model.device
print(device)

# -1 = embed all files
total_files_to_embed = -1

def file_reader(directory, filename):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', errors='ignore') as file:
            content = file.read()

            return content
        
    elif filename.endswith('.pdf'):
        reader = PdfReader(os.path.join(directory, filename))
        all_text = ''
        for page in reader.pages:
            all_text += page.extract_text() + ' '
        
        return all_text

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
        content = file_reader(directory, filename)

        documents = encode_document(
            filename, 
            documents, 
            args.add_file_content, 
            content, 
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
                
    return documents

# Example usage
documents = load_and_preprocess_text_files(args.directory_path)

# Save documents with embeddings to a JSON file
with open(os.path.join('..', 'data', args.index_file_name), 'w') as f:
    json.dump(documents, f)