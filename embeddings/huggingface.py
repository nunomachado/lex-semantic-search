import torch
import numpy as np

from typing import List
from transformers import AutoTokenizer, AutoModel
from embeddings.encoder import EmbeddingEncoder


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


class HuggingFaceEncoder(EmbeddingEncoder):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = cls_pooling(model_output)

        return embeddings
