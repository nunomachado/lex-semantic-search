import pandas as pd
import numpy as np

from datasets import load_dataset


def convert_embeddings(embeddings_str):
    embeddings = [np.fromstring(e[1:-1], sep=' ') for e in embeddings_str]
    return embeddings


class DatasetLoader:
    @staticmethod
    def load_from_file(path: str, to_pandas=False):
        if to_pandas:
            return pd.read_csv(path)
        else:
            return load_dataset("csv", data_files=path)['train']

    @staticmethod
    def load_from_file_with_embeddings(path: str, to_pandas=False):
        dataset = load_dataset("csv", data_files=path)['train']
        new_dataset = dataset.map(
            lambda example: {
                'embeddings': convert_embeddings(example['embeddings'])
            },
            batched=True,
        )
        if to_pandas:
            return new_dataset.to_pandas()
        else:
            return new_dataset

    @staticmethod
    def load_from_huggingface(dataset_name: str, split: str, to_pandas=False):
        # load dataset from HuggingFace hub
        dataset = load_dataset(dataset_name, split=split)
        if to_pandas:
            return dataset.to_pandas()
        else:
            return dataset
