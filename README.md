# Offline NLP exam assistant

A local Streamlit app that retrieves relevant course passages and drafts answers
with Qwen through Ollama. Course materials and downloaded model weights are not
included. Add your own permitted materials to `data/`.

## Windows setup with Anaconda

Install Python through Anaconda and install [Ollama](https://ollama.com/download/windows).
Clone this repository, open Anaconda Prompt in its directory, and run:

```bat
conda create -n rag-nlp python=3.11 -y
conda activate rag-nlp
python -m pip install -r requirements.txt
python -c "from pathlib import Path; from sentence_transformers import SentenceTransformer; Path('index').mkdir(exist_ok=True); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('index/embedding-model')"
ollama pull qwen2.5:3b
```

Installation and model downloads require internet once. Reopen your terminal if
Windows does not recognize `ollama` after installing it.

## Use

1. Put your PDFs, DOCX documents, PPTX slides, or IPYNB notebooks in `data/`.
2. Start Ollama, then run from the repository directory:

```bat
conda activate rag-nlp
python -m streamlit run app.py
```

Open the local address printed in the terminal (usually http://localhost:8501).
The first run builds the course index. Ask a question and choose **Draft answer**.
Expand the evidence to check the quoted sources. Subsequent use can be offline
once both models are downloaded. Keep the `index/` folder on your computer.

Embedding weights stay cached while Streamlit runs. The disk index is reused
when source files are unchanged. Qwen stays loaded after an answer; free its
memory when finished with `ollama stop qwen2.5:3b`.

## Limitations

Answers can be incorrect: check the retrieved sources. Extraction reads document
text, not scanned pages, slide images, or notebook output images. The current
app cites passage numbers rather than exact page locations. Empty data folders
cannot be indexed. Use during exams only where permitted.

## Local checkpoints

Before editing source or configuration, run:

```bat
python checkpoint.py "before my change"
```

After editing, run it again with a reason starting with `after`. Checkpoints
remain local and are excluded from Git along with course data, models, and indexes.
