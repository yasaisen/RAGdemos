# tryRAG
A small exploration of retrieval-augmented generation (RAG) in Python.

## Repository Layout

- `tryRAG/framework.py` – main class with indexing and generation logic.
- `tryRAG/dataType.py` – dataclasses for document chunks and query objects.
- `tryRAG/templates.py` – prompt templates used by the framework.
- `demo.ipynb` – Jupyter notebook that demonstrates basic usage.

## Installation
1. Create a Python environment (tested with Python 3.10):
   ```bash
   conda create --name tryRAG python=3.10
   conda activate tryRAG
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yasaisen/tryRAG.git
   ```
3. Install the dependencies:
   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
   pip install transformers==4.51.3 accelerate==0.26.0 faiss-cpu==1.11.0.post1 sentence-transformers==5.0.0
   ```

## Usage

1. Prepare a JSON lines file where each line contains a document with `text` and `url` fields.
2. Create the framework and load your documents:

```python
from tryRAG.framework import RAGFramework

cfg = {
    "lm_model_name": "google/gemma-3-4b-it",
    "emb_model_name": "all-MiniLM-L6-v2",
    "doc_path": "path/to/data.jsonl", 
    # "idx_path": "path/to/saved/dir",
    "device": "cuda",
}
rag = RAGFramework.from_config(cfg)

```

3. Ask a question:

```python
response = rag.ask("What courses are offered in EECS?")
print(response["response"])
```

The framework retrieves the most relevant document chunks, builds a prompt, and
uses a language model to produce an answer.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
