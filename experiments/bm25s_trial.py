"""
Script to experiment with BM25s.
https://pypi.org/project/bm25s/
https://bm25s.github.io/
"""

import bm25s

corpus = [
    'this is a test script',
    'this library is amazing and allows to chat with 1000s of text files',
    'can up text files, pdfs, images, and videos as well'
]

# Tokenize.
corpus_tokens = bm25s.tokenize(corpus)
print(f"Tokenized corpus: {corpus_tokens}")

retriever = bm25s.BM25(corpus=corpus)
# Index the corpus.
retriever.index(corpus_tokens)

# Search.
query = 'is this library amazing?'
query_tokens = bm25s.tokenize(query)
results, scores = retriever.retrieve(query_tokens=query_tokens, k=2)

print(results)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Doc: {doc}, score: {score}")