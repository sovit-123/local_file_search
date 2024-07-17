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

FURTHER NOTE:
Recommended to use this script to create embeddings rather than 
`create_embeddings_no_cleaning.py`. This cleans up useless spaces and 
numbers which creates a more meaningful and compact embedding. The results
for simple phrases will be more or less the same. However, for complex
queries, the embeddings from this script will generate better answers.


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
import argparse

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from joblib import Parallel, delayed

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
args = parser.parse_args()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

# -1 = embed all files
total_files_to_embed = -1

# Load SpaCy model
if device == 'cuda' or device == 'cuda:0':
    spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')

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
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', errors='ignore') as file:
            content = file.read()
            preprocessed_content = preprocess_text(content)
            # print(len(preprocessed_content), len(content))
            # print(preprocessed_content)
            documents = encode_document(
                filename, 
                documents,
                args.add_file_content, 
                content, 
                preprocessed_content,
                chunk_size=512,
                overlap=50 
            )
            return documents
            
    return None

if __name__ == '__main__':
    results = []

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