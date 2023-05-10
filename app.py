import gradio as gr

from utils.dataset_loader import DatasetLoader
from embeddings.huggingface import HuggingFaceEncoder
from search.faiss import FaissSearchEngine

# Preload dataset with embeddings for chunk_size = 25
ds_test_embeddings = DatasetLoader.load_from_file_with_embeddings("./data/df_chunked_25_with_embeddings.csv")
hf_encoder = HuggingFaceEncoder("sentence-transformers/multi-qa-mpnet-base-dot-v1")


def retrieve_chunks(query, chunk_size, embeddings_generator, retriever_method, num_chunks_to_retrieve):
    # Ignore chunk_size, embeddings_generator, and retriever_method,
    # as we currently support only a single configuration
    faiss_search = FaissSearchEngine(ds_test_embeddings, hf_encoder)

    return faiss_search.search(query, num_chunks_to_retrieve)


# Create the Gradio application
with gr.Blocks() as demo:
    query = gr.inputs.Textbox(label='Query', placeholder="Enter your query here. Example: 'What is a transformer?'")
    chunk_size = gr.inputs.Slider(
        minimum=25,
        maximum=25,
        step=25,
        default=25,
        label='Chunk Size'
    )
    embeddings_generator = gr.Radio(
        ['sentence-transformers/multi-qa-mpnet-base-dot-v1'],
        label='Embeddings Generator',
        value='sentence-transformers/multi-qa-mpnet-base-dot-v1'
    )
    retriever_method = gr.Radio(
        ['FAISS'],
        value="FAISS",
        label="Retriever Method"
    )
    num_chunks_to_retrieve = gr.inputs.Slider(
        minimum=3,
        maximum=5,
        step=1,
        default=3,
        label='Number of Chunks to Retrieve'
    )
    inputs = [query, chunk_size, embeddings_generator, retriever_method, num_chunks_to_retrieve]

    submit_btn = gr.Button("Submit")

    outputs = gr.Dataframe(
        headers=['id', 'guest', 'title', 'text', 'start', 'end', 'scores'],
        type="pandas",
        wrap=True
    )

    submit_btn.click(retrieve_chunks, inputs=inputs, outputs=outputs)

# Run the Gradio application
demo.launch()
