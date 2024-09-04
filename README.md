# Local File Search using Embeddings

Scripts to replicate simple file search and RAG in a directory with embeddings and Language Models.

From scratch implementation, ***no Vector DBs yet.***

***A simplified use case***: You have thousands of research papers but don't know which are the ones containing content that you want. You do a search according to a rough query and get an adequately good results. 

## Setup

Run the following in terminal in your preferred virtual/conda environment.

```
sh setup.sh
```

It will install the the requirements from the `requirements.txt` file.

## Updates

* September 4, 2024: Added image, PDF, and text file chat to `ui.py` with multiple Phi model options.

* September 1, 2024: Now you can upload PDFs directly to the Gradio UI (`python ui.py`) and start chatting.

## Steps to Chat with Any PDF in Graio UI

<span style="color: green">***You can run `ui.py` and select the any PDF file in the Gradio UI to interactively chat with the document.***</span> <span style="color: purple">***(Just do `python ui.py` and start chatting)***</span>

## Steps to Run Through CLI

* (**Optional**) Download the `papers.csv` file from [here](https://www.kaggle.com/datasets/benhamner/nips-papers?select=papers.csv) and keep in the `data` directory. <span style="color: red">**You can also keep PDF files in the directory and pass the directory path**.</span>

* (**Optional**) <span style="color: red">*Execute this step only if you download the above CSV file. Not needed if you have your own text files or PDFs in a directory*</span>. Run the `csv_to_text_files.py` script to generate a directory of text files from the CSV file. 

* Run `create_embeddings.py` to generate the embeddings that are stored in JSON file in the `data` directory. Check the scripts for the respective file names. ***Check `src/create_embedding.py`*** for relevant command line arguments to be passed.

  * Generate example:

    ```
    python create_embeddings.py --index-file-name index_file_to_store_embeddings.json --directory-path path/to/directory/containing/files/to/embed
    ```

  * Additional command line arguments:

    * `--add-file-content`: To store text chunks in JSON file if planning to do RAG doing file file search.
    * `--model`: Any [Sentence Transformer](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) model tag. Default is `all-MiniLM-L6-v2`.
    * `--chunk-size` and `--overlap`: Chunk size for creating embeddings and overlap between chunks.
    * `--njobs`: Number of parallel processes to use. Useful when creating embeddings for hundreds of files in a directory.

* Then run the `search.py` with the path to the respective embedding file to start the search. Type in the search query.

  * General example:

    ```
    python search.py --index-file path/to/index.json
    ```
  
    The above command just throws a list of TopK files that matches the query.

  * Additional command line arguments:

    * `--extract-content`: Whether to print the related content or not. Only works if `--add-file-content` was passed during creation of embeddings.
    * `--model`: Sentence Transformer model tag if a model other than `all-MiniLM-L6-v2` was used during the creation of embeddings.
    * `--topk`: Top K embeddings to match and output to the user.
    * `--llm-call`: Use an LLM to restructure the answer for the question asked. Only works if `--extract-content` is passed as the model will need context. Currently the Phi-3 Mini 4K model is used. 
  


## Datasets

* [NIPS Research Papers](https://www.kaggle.com/datasets/benhamner/nips-papers?select=papers.csv)
