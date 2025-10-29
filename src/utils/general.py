from transformers import TextStreamer

import requests
import os
import urllib.request

YELLOW = "\033[93m"
RESET = "\033[0m"

class MyTextStreamer(TextStreamer):
    """
    This class is mainly used to override the methods for text streamer 
    stdout and apply a different color for printing the text.
    """
    # Override method.
    def on_finalized_text(self, text, stream_end = False):
        """
        Prints the new text to stdout. 
        If the stream is ending, also prints a newline.
        """
        # print(text, flush=True, end="" if not stream_end else None)
        print(f"{YELLOW}{text}{RESET}", flush=True, end="" if not stream_end else None)

def download_arxiv_doc(url):
    # url = 'https://arxiv.org/pdf/2104.14294'

    download_dir = os.path.join('..', 'data', 'downloaded_arxiv')

    os.makedirs(download_dir, exist_ok=True)

    name = url.split('/')[-1]

    download_path = os.path.join(download_dir, name+'.pdf')

    response = requests.get(url)

    with open(download_path, 'wb') as f:
        f.write(response.content)

    return download_path

def download_md(url):
    download_dir = os.path.join('..', 'data', 'downloaded_md')

    os.makedirs(download_dir, exist_ok=True)

    name = url.split('/')[-1]

    download_path = os.path.join(download_dir, name+'.md')

    response = requests.get(url+'.md')

    with open(download_path, 'wb') as f:
        f.write(response.content)

    return download_path

def download_wiki_doc(url):
    """
    Download a Wikipedia page as PDF using Wikipedia's built-in PDF export API.
    """
    base_url = 'https://en.wikipedia.org/api/rest_v1/page/pdf/'

    title = url.split('/')[-1]
    final_url = base_url + title.replace(' ', '_')

    print(final_url)

    # Add a User-Agent header to avoid 403 Forbidden
    req = urllib.request.Request(
        final_url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                               'AppleWebKit/537.36 (KHTML, like Gecko) '
                               'Chrome/120.0 Safari/537.36'}
    )

    with urllib.request.urlopen(req) as response:
        pdf_data = response.read()

    download_dir = os.path.join('..', 'data', 'downloaded_wiki')

    os.makedirs(download_dir, exist_ok=True)

    download_path = os.path.join(download_dir, title+'.pdf')

    with open(download_path, 'wb') as f:
        f.write(pdf_data)

    return download_path