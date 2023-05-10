import pandas as pd

from datasets import Dataset
from embeddings.encoder import EmbeddingEncoder


class FaissSearchEngine:
    def __init__(self, embeddings: Dataset, encoder: EmbeddingEncoder):
        self.embeddings = embeddings
        # assume dataset has a column "embeddings" with the embeddings
        self.embeddings.add_faiss_index(column="embeddings")
        self.encoder = encoder

    def search(self, query, k=5):
        # Encode the query using the same model that was used to generate the embeddings
        query_embedding = self.encoder.generate_embeddings(query).numpy()

        # Search the index using FAISS
        scores, samples = self.embeddings.get_nearest_examples("embeddings", query_embedding, k)

        # Return the results as a list of dictionaries
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        #samples_df.sort_values("scores", ascending=False, inplace=True)
        samples_df = samples_df.drop("embeddings", axis=1)

        return samples_df
