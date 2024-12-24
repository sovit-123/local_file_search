from transformers import TextStreamer

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