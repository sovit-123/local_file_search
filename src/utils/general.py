from transformers import TextStreamer

import requests
import os

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