# Local File Search using Embeddings

Scripts to replicate simple file search in a directory with embeddings.

This is an oversimplified process just to semantically search the file names that may contain the search query. ***No VectorDB, No LLMs (yet)***.

A simplified use case: You have thousands of research papers but don't know which are the ones containing content that you want. You do a search according to a rough query and get an adequately good results.

## Setup

Run the following in terminal in your preferred virtual/conda environment.

```
sh setup.sh
```

It will install the the requirements from the `requirements.txt` file and download the Spacy model.

## Steps to Run

* Download the `papers.csv` file from [here](https://www.kaggle.com/datasets/benhamner/nips-papers?select=papers.csv) and keep in the `data` directory.
* Run the `csv_to_text_files.py` script to generate a directory of text files from the CSV file.
* Run either `create_embeddings.py` or `create_embeddings_no_cleaning.py` to generate the embeddings that are stored in JSON file in the `data` directory. Check the scripts for the respective file names.
  ***Note: Recommended to the `create_embeddings.py` as this cleans the numbers and white spaces using Spacy. Generates slightly better results when searching.***
* Then run the `search.py` with the path to the respective embedding file to start the search. Type in the search query.

## Datasets

* [NIPS Research Papers](https://www.kaggle.com/datasets/benhamner/nips-papers?select=papers.csv)
