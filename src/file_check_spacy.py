"""
Minimal script to check what SpaCy does to a file with rubbish words.
"""

import spacy

spacy.prefer_gpu()

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

with open('../data/paper_files/file_797.txt', 'r') as file:
    text = file.read()

file.close()

print(text)

print('#'*50)

doc = nlp(text)

tokens = [
    token.text for token in doc if \
    # not token.is_stop \
    # and not token.is_punct \
    not token.is_space \
    and not token.is_digit
]

print(' '.join(tokens))