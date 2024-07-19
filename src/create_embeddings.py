"""
Script to create SBERT embedding and store in JSON file. 
We clean the text using a SpaCy model here.

This gives slightly worse results in majority of the cases compared to the
embeddings generated using `create_embeddings_no_cleaning.py` when doing
cosine similarity search on chunks of text.

Requirements:
$ pip install spacy
$ python -m spacy download en_core_web_sm

USAGE:
$ python create_embeddings.py --index-file-name my_index_file.json
"""

import os
import spacy
import torch
import json
import argparse
import multiprocessing

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from pypdf import PdfReader

multiprocessing.set_start_method('spawn', force=True)

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
    '--njobs',
    default=8,
    help='number of parallel processes to use'
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cuda'
print(device)

# -1 = embed all files
total_files_to_embed = -1

# Load SpaCy model
if device == 'cuda' or device == 'cuda:0':
    spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')

# Load SBERT model
model_id = args.model
# model_id = 'outputs/checkpoint-12500' # Or any other custom model path.
model = SentenceTransformer(model_id).to(device)
print(f"SBERT model device: {next(model.parameters()).device}")

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

def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.text for token in doc if \
        not token.is_space \
    ]
    processed_text = ' '.join(tokens)
    return processed_text

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
    filename, 
    documents,
    add_file_content, 
    content,
    preprocessed_content, 
    chunk_size=512, 
    overlap=50
):
    """Encode the document in chunks."""
    chunks = chunk_text(preprocessed_content, chunk_size, overlap)

    if not chunks: # If no chunks are possible.
        features = extract_features(preprocessed_content).tolist()
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

def load_and_preprocess_text_files(directory, filename, documents):
    content = file_reader(directory, filename)
    preprocessed_content = preprocess_text(content)

    documents = encode_document(
        filename, 
        documents,
        args.add_file_content, 
        content, 
        preprocessed_content,
        chunk_size=args.chunk_size,
        overlap=args.overlap 
    )
    return documents
            
if __name__ == '__main__':
    results = []
    
    directory = args.directory_path
    all_files = os.listdir(directory)
    all_files.sort()
    if total_files_to_embed > -1:
        files_to_embed = all_files[:total_files_to_embed]
    else: 
        files_to_embed = all_files
    
    results = Parallel(
        n_jobs=args.njobs, 
        backend='multiprocessing'
    )(delayed(load_and_preprocess_text_files)(directory, filename, results) \
            for filename in tqdm(files_to_embed, total=len(files_to_embed))
        )
    
    documents = [res for result in results for res in result]
    
    # Save documents with embeddings to a JSON file
    with open(os.path.join('..', 'data', args.index_file_name), 'w') as f:
        json.dump(documents, f)