"""
Script to conver the Enron email dataset CSV content to text files.
"""

import os
import pandas as pd

from tqdm.auto import tqdm

os.makedirs('../data/paper_files/', exist_ok=True)

df = pd.read_csv('../data/papers.csv')

num_samples_to_generate = -1

for i, text in tqdm(enumerate(df['paper_text'].tolist()), total=len(df['paper_text'].tolist())):
    if num_samples_to_generate != -1 and i == num_samples_to_generate:
        break
    with open(f"../data/paper_files/file_{df['id'][i]}.txt", 'w') as f:
        # if 'computer vision' in text or 'Computer Vision' in text or 'Computer vision' in text:
        #     print(df['id'][i])
        f.writelines(text)