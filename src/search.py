"""
Search based on cosine similarity.

USAGE:
$ python search.py --index-file path/to/index.json
"""

import json
import argparse
import torch
import os

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from llm import generate_next_tokens
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from utils.general import MyTextStreamer
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--index-file',
        dest='index_file',
        # required=True,
        help='path to an indexed embedding JSON file'
    )
    parser.add_argument(
        '--extract-content',
        dest='extract_content',
        action='store_true',
        help='whether to print the related content or not \
            as the index file does not always contain file content'
    )
    parser.add_argument(
        '--model',
        default='all-MiniLM-L6-v2',
        help='embedding model id from hugging face'
    )
    parser.add_argument(
        '--llm-call',
        dest='llm_call',
        action='store_true',
        help='make call to an llm to restructure the answer'
    )
    parser.add_argument(
        '--topk',
        default=5,
        type=int,
        help='number of chunks to retrieve'
    )
    parser.add_argument(
        '--web-search',
        dest='web_search',
        action='store_true',
        help='do a web search to get context, uses Tavily API key'
    )
    args = parser.parse_args()
    return args

def load_embedding_model(model_id=None):
    # Load SBERT model
    model = SentenceTransformer(model_id)
    return model

def extract_features(text, model):
    """Generate SBERT embeddings for the input text."""
    return model.encode(text)

def process_query(query, model):
    """Preprocess the query and generate SBERT embeddings."""
    query_features = extract_features(query, model).tolist()
    return query_features

def search(query, documents, model, top_k=5):
    """Search for the most relevant documents to the query."""
    print('SEARCHING...')
    query_features = process_query(query, model)
    scores = []
    for document in tqdm(documents, total=len(documents)):
        score = model.similarity([query_features], [document['features']])[0][0]
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

def extract_relevant_part(query, content, model, chunk_size=32, overlap=4):
    """Extract the part of the content that is most relevant to the query."""
    chunks = chunk_text(content, chunk_size, overlap)
    if not chunks:
        return content  # Return full content if it can't be split

    chunk_embeddings = model.encode(chunks)
    query_embedding = extract_features(query, model)
    scores = model.similarity([query_embedding], chunk_embeddings).flatten()
    best_chunk_idx = scores.argmax()
    return chunks[best_chunk_idx]


def load_documents(file_path):
    """Load preprocessed documents and embeddings from a JSON file."""
    with open(file_path, 'r') as f:
        documents = json.load(f)
    return documents


def do_web_search(query=None):
    """
    Do a web search using Tavily to get the context and return to the model.

    :param query: Search query.

    Returns:
        retrieved_docs: a list of retrieved web docs/a list of string results.
            e.g. ['context 1', 'context 2']
    """
    assert TAVILY_API_KEY is not None, 'TAVILY_API_KEY not found, please check your .env file'

    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(query)

    results = [res['content'] for res in response['results']]
    return results


def main(documents, query, model, extract_content, topk=5, web_search=False):
    RED = "\033[31m"
    RESET = "\033[0m"
    # Perform search.
    if web_search:
        results = do_web_search(query=query)
        return results
    else:
        results = search(query, documents, model, topk)
    
    relevant_parts = []
    retrieved_docs = []
    for result in results:
        document = result[0]
        print(f"Filename: {result[0]['filename']}, Score: {result[1]}")
        # Search for relevevant content if `--extract-content` is passed.
        if extract_content:
            try:
                document['content']
                retrieved_docs.append(document['content'])
            except:
                raise AssertionError(f"It looks like you have passed "
                f"`--extract-content` but the document does not contain "
                f"original file content. Please check again... "
                f"Either create a new index file with the file content or "
                f"remove `--extract-content` while executing the search script"
                )
            
            relevant_part = extract_relevant_part(query, document['content'], model)
            relevant_parts.append(relevant_part)
            # Few color modifications to make the output more legible.
            document['content'] = document['content'].replace(relevant_part, f"{RED}{relevant_part}{RESET}")
            print(f"Retrieved document: {document['content']}\n")

    return retrieved_docs

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_id = 'microsoft/Phi-4-mini-instruct'

    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    args = parser_opt()
    topk = args.topk
    extract_content = args.extract_content

    # Load embedding model.
    embedding_model = load_embedding_model(args.model)
    
    # Load documents if user does not demand web search.
    documents_file_path = args.index_file
    if not args.web_search:
        documents = load_documents(documents_file_path)
    else:
        documents = None

    # Load the LLM only when if `args.llm` has been passed by user.
    if args.llm_call:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map=device,
            trust_remote_code=True
        )
        streamer = MyTextStreamer(
            tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    # Keep on asking the user prompt until the user exits.
    while True:
        query = input(f"\n{RED}Enter your search query:{RESET} ")
        context_list = main(
            documents, 
            query, 
            embedding_model, 
            extract_content, 
            topk, 
            web_search=args.web_search
        )
    
        if args.llm_call:
            print(f"\n{GREEN}Generating LLM response...\n{RESET}")
            context = '\n\n'.join(context_list)
        
            generate_next_tokens(
                user_input=query, 
                context=context,
                model=llm_model,
                tokenizer=tokenizer,
                streamer=streamer,
                device=device,
                model_id=model_id
            )