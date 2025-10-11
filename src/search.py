"""
Search based on cosine similarity.

USAGE:
$ python search.py --index-file path/to/index.json
"""

import json
import argparse
import torch
import os

from tqdm import tqdm
from llm import generate_next_tokens
from utils.general import MyTextStreamer
from utils.load_models import load_embedding_model, load_llm
from dotenv import load_dotenv
from tavily import TavilyClient
from perplexity import Perplexity

load_dotenv()

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

class DenseSearch():
    """
    Class for implementing dense search along with extracting features
    from text using embedding model, processing query, and returning
    the dense search cosine scores.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def extract_features(self, text):
        """Generate SBERT embeddings for the input text."""

        return self.embedding_model.encode(text)
    
    def process_query(self, query):
        """Preprocess the query and generate SBERT embeddings."""
        query_features = self.extract_features(query).tolist()

        return query_features
    
    def chunk_text(self, text, chunk_size=100, overlap=50):
        """Chunk the text into overlapping windows."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break

        return chunks
    
    def extract_relevant_part(self, query, content, chunk_size=32, overlap=4):
        """Extract the part of the content that is most relevant to the query."""
        chunks = self.chunk_text(content, chunk_size, overlap)
        if not chunks:
            return content  # Return full content if it can't be split

        chunk_embeddings = self.embedding_model.encode(chunks)
        query_embedding = self.extract_features(query)
        scores = self.embedding_model.similarity([query_embedding], chunk_embeddings).flatten()
        best_chunk_idx = scores.argmax()

        return chunks[best_chunk_idx]
    
    def search(self, query, documents, top_k=5):
        """
        Dense search for the most relevant documents to the query.
        Similarity score => cosine similarity. 
        """
        print('SEARCHING...')
        query_features = self.process_query(query)
        scores = []
        for document in tqdm(documents, total=len(documents)):
            score = self.embedding_model.similarity([query_features], [document['features']])[0][0]
            scores.append((document, score))
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


def load_documents(file_path):
    """Load preprocessed documents and embeddings from a JSON file."""
    with open(file_path, 'r') as f:
        documents = json.load(f)
    return documents


def do_web_search(query=None, search_engine='perplexity'):
    """
    Do a web search using Tavily to get the context and return to the model.

    :param query: search query
    :param search_engine: search engine to use, either 'tavily' or 'perplexity'

    Returns:
        retrieved_docs: a list of retrieved web docs/a list of string results.
            e.g. ['context 1', 'context 2']
    """
    if search_engine == 'tavily':
        TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
        assert TAVILY_API_KEY is not None, 'TAVILY_API_KEY not found, please check your .env file'
    
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query)
    
        results = [res['content'] for res in response['results']]
    elif search_engine == 'perplexity':
        PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
        assert PERPLEXITY_API_KEY is not None, 'PERPLEXITY_API_KEY not found, please check your .env file'

        ppxl_client = Perplexity()
        response = ppxl_client.search.create(
            query=query,
            max_results=5,
            max_tokens_per_page=512
        )

        results = [result.snippet for result in response.results]

    return results


def main(
    documents, 
    query, 
    model, 
    extract_content, 
    topk=5, 
    web_search=False, 
    search_engine='perplexity',
    dense_searcher=None
):
    RED = "\033[31m"
    RESET = "\033[0m"

    if web_search: # Perform search.
        results = do_web_search(query=query, search_engine=search_engine)
        return results
    else: # Perform dense similrity search.
        if dense_searcher is None:
            dense_searcher = DenseSearch(model)
        results = dense_searcher.search(query, documents, topk)
    
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
            
            relevant_part = dense_searcher.extract_relevant_part(query, document['content'])
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
        llm_model, tokenizer = load_llm(model_id=model_id, device=device)
        streamer = MyTextStreamer(
            tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    # Load dense search engine.
    dense_searcher = DenseSearch(embedding_model=embedding_model)

    # Keep on asking the user prompt until the user exits.
    while True:
        query = input(f"\n{RED}Enter your search query:{RESET} ")
        context_list = main(
            documents, 
            query, 
            embedding_model, 
            extract_content, 
            topk, 
            web_search=args.web_search,
            dense_searcher=dense_searcher
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