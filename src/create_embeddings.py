"""
This script creates embeddings for text files in a directory and stores them 
in a JSON file.

USAGE:
$ python create_embeddings.py --index-file-name my_index_file.json
"""

import os
import json
import argparse
import multiprocessing
import glob as glob

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from pypdf import PdfReader
from joblib import Parallel, delayed

multiprocessing.set_start_method('spawn', force=True)

def parse_opt():
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
        default=128,
        type=int,
        help='chunk size of embedding creation and extracing content if needed'
    )
    parser.add_argument(
        '--overlap',
        default=16,
        type=int,
        help='text overlap when creating chunks'
    )
    parser.add_argument(
        '--njobs',
        default=8,
        help='number of parallel processes to use'
    )
    args = parser.parse_args()
    return args

# Load SBERT model
def load_model(args):
    model_id = args.model
    model = SentenceTransformer(model_id)
    # Device setup (not needed for SentenceTransformer as it handles it internally)
    device = model.device
    print(device)
    return model

# -1 = embed all files
total_files_to_embed = -1

def file_reader(filename):
    if filename.endswith('.txt'):
        with open(os.path.join(filename), 'r', errors='ignore') as file:
            content = file.read()

            return content
        
    elif filename.endswith('.pdf'):
        reader = PdfReader(os.path.join(filename))
        all_text = ''
        for page in reader.pages:
            all_text += page.extract_text() + ' '
        
        return all_text

def extract_features(text, model):
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
    chunk_size=512, 
    overlap=50,
    model=None
):
    """Encode the document in chunks."""
    chunks = chunk_text(content, chunk_size, overlap)

    if not chunks:  # If no chunks are possible.
        features = extract_features(content, model).tolist()
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
            features = extract_features(chunk, model).tolist()
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

def load_and_preprocess_text_files(documents, filename, args, model=None):
    """
    Loads and preprocesses text files in a directory.

    :param directory: The directory containing the text files.

    Returns:
        documents: A list of dictionaries containing filename and embeddings.
    """
    content = file_reader(filename)

    documents = encode_document(
        filename, 
        documents, 
        args.add_file_content, 
        content, 
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model=model
    )
                
    return documents

if __name__ == '__main__':
    args = parse_opt()

    model = load_model(args)

    results = []

    all_files = glob.glob(os.path.join(args.directory_path, '**'), recursive=True)
    all_files = [filename for filename in all_files if not os.path.isdir(filename)]
    print(all_files)
    all_files.sort()
    if total_files_to_embed > -1:
        files_to_embed = all_files[:total_files_to_embed]
    else:
        files_to_embed = all_files

    results = Parallel(
        n_jobs=args.njobs, 
        backend='multiprocessing'
    )(delayed(load_and_preprocess_text_files)(results, filename, args, model) \
            for filename in tqdm(files_to_embed, total=len(files_to_embed))
        )
    
    documents = [res for result in results for res in result]
    
    # Save documents with embeddings to a JSON file
    with open(os.path.join('..', 'data', args.index_file_name), 'w') as f:
        json.dump(documents, f)