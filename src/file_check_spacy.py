"""
Minimal script to check file manipulation with Spacy.
"""

import spacy

spacy.prefer_gpu()

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

with open('../data/paper_files/file_1.txt', 'r') as file:
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
    # and not token.is_digit
]

final_text = ' '.join(tokens) 

print(final_text)

print(len(text), len(final_text))