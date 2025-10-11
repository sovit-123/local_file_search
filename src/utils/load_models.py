"""
Script to load embedding and LLM models for `search.py`
"""

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)

def load_embedding_model(model_id=None):
    """
    Loading embedding model from Sentence Transformers.

    :param model_id: model_id/tag from Hugging Face Sentence Transformers.

    Returns:
        model: The embedding model.
    """
    model = SentenceTransformer(model_id)

    return model


def load_llm(model_id, device='cuda:0'):
    """
    Loading LLMs from Hugging Face.

    :param model_id: model_id/tag from Hugging Face model Hub.

    Returns:
        llm_model: The loaded LLM.
        tokenizer: The LLM's tokenizer.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=device,
        trust_remote_code=True
    )

    return model, tokenizer