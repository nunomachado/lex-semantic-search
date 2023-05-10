from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingEncoder(ABC):
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        pass
