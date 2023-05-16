---
title: Lex Fridman Podcast Semantic Search
emoji: 💡
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 3.28.3
app_file: app.py
pinned: false
---
# lex-semantic-search

Gradio application for performing semantic search on Lex Fridman podcast transcripts.

## Dataset

The Gradio application is pre-loaded with chunks (chunk size is 25 contiguous entries) and embeddings for dataset [nmac/lex_fridman_podcast](https://huggingface.co/datasets/nmac/lex_fridman_podcast).

## Usage

1. Set up virtual environment with the required dependencies:
```bash
python -m venv lex-semantic-search
source lex-semantic-search/bin/activate 
pip install -r requirements_cpu.txt # for CPU
pip install -r requirements_gpu.txt # for GPU
```

2. Run the application locally using the following command:
```bash
python app.py
```

3. Access the application by opening your web browser and navigating to http://localhost:7860.

4. In the application interface, adjust the input settings according to your needs:
   - **Query:** Enter a query to search for relevant podcast transcript chunks related to it.
   - **Chunk Size:** Adjust the chunk size. *(Fixed to 25)*
   - **Embeddings Generator:** Select the embeddings generator to use. *(Fixed to `sentence-transformers/multi-qa-mpnet-base-dot-v1`)*
   - **Retriever Method:** Select the retriever method. *(Fixed to `FAISS`)*
   - **Number of Chunks to Retrieve:** Set the number of chunks to retrieve.

5. Click the "Submit" button to retrieve the chunks that match your settings and query. The results will be displayed in a table.
