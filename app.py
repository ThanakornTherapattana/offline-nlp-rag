import json
import os
import re
from pathlib import Path

import numpy as np
import requests
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"
INDEX = ROOT / "index"
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def extract(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if ext == ".docx":
        from docx import Document
        d = Document(path)
        return "\n".join([p.text for p in d.paragraphs] + [" ".join(c.text for row in t.rows for c in row.cells) for t in d.tables])
    if ext == ".pptx":
        from pptx import Presentation
        out = []
        for n, slide in enumerate(Presentation(path).slides, 1):
            out.append(f"Slide {n}: " + "\n".join(s.text for s in slide.shapes if s.has_text_frame))
        return "\n".join(out)
    if ext == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") in ("markdown", "code"))
    return ""


def chunks(text, size=850, overlap=130):
    text = re.sub(r"\s+", " ", text).strip()
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(". ", start + size // 2, end)
            if boundary > start: end = boundary + 1
        if text[start:end].strip(): out.append(text[start:end].strip())
        start = max(start + 1, end - overlap)
    return out


@st.cache_resource(show_spinner="Loading embedding weights once for this app process...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    INDEX.mkdir(exist_ok=True)
    embedding_path = INDEX / "embedding-model"
    model_name = str(embedding_path) if embedding_path.exists() else "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name, local_files_only=True)


def course_signature():
    files = sorted(p for p in DATA.rglob("*") if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".ipynb"})
    # JSON round-trips lists, not tuples. Keep the same type on both sides.
    return [[str(p.relative_to(DATA)), p.stat().st_size, p.stat().st_mtime_ns] for p in files]


@st.cache_resource(show_spinner="Loading course index...", max_entries=1)
def build_index(signature):
    INDEX.mkdir(exist_ok=True)
    meta_file, vec_file = INDEX / "chunks.json", INDEX / "vectors.npy"
    stamp = INDEX / "signature.json"
    if meta_file.exists() and vec_file.exists() and stamp.exists() and json.loads(stamp.read_text()) == signature:
        return json.loads(meta_file.read_text(encoding="utf-8")), np.load(vec_file)
    records = []
    for name, _, _ in signature:
        p = DATA / name
        try:
            for i, text in enumerate(chunks(extract(p))):
                records.append({"source": p.name, "chunk": i + 1, "text": text})
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
    if not records:
        raise ValueError("No readable passages found in the data folder.")
    vectors = load_embedder().encode([r["text"] for r in records], normalize_embeddings=True, show_progress_bar=False)
    meta_file.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    np.save(vec_file, vectors)
    stamp.write_text(json.dumps(signature), encoding="utf-8")
    return records, vectors


st.set_page_config(page_title="Offline NLP Exam RAG", page_icon="📚", layout="wide")
st.title("Offline NLP Exam Assistant")
st.caption("Course-material retrieval with local answer drafting. After setup, questions and documents stay on this computer.")
with st.sidebar:
    st.subheader("Local setup")
    st.write(f"Answer model: `{MODEL}` via Ollama")
    st.write("Embeddings: `all-MiniLM-L6-v2` (local)")
    st.write(f"Course files: `{DATA}`")
    if st.button("Rebuild course index"):
        (INDEX / "signature.json").unlink(missing_ok=True)
        build_index.clear()
        st.rerun()
    st.caption("Embedding weights stay cached while Streamlit runs. Qwen stays loaded after the first answer until you stop it or Ollama restarts.")

try:
    embedder = load_embedder()
    records, vectors = build_index(course_signature())
except Exception as e:
    st.error(f"Local index/model unavailable: {e}")
    st.info("Run setup.bat once while online to install dependencies and download the embedding model, then start Ollama with qwen2.5:3b.")
    # st.stop() can be ineffective when the file is executed outside Streamlit.
    # Raising prevents later code from referencing an unassigned index.
    raise SystemExit(1)

st.success(f"Indexed {len(records)} passages from {len(set(r['source'] for r in records))} course files.")
question = st.text_area("Exam question", height=120, placeholder="Paste or type a question...")
top_k = st.slider("Evidence passages", 2, 8, 5)
if st.button("Draft answer", type="primary", disabled=not question.strip()):
    qv = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ qv
    ids = np.argsort(scores)[::-1][:top_k]
    evidence = "\n\n".join(f"[S{i+1}] {records[j]['source']} (passage {records[j]['chunk']}): {records[j]['text']}" for i, j in enumerate(ids))
    prompt = f"Answer the student's NLP exam question using only the supplied course evidence. If evidence is insufficient, say so. Be concise but explanatory; define terms and show steps where relevant. Cite claims inline as [S1], [S2]. Do not invent facts.\n\nQUESTION:\n{question}\n\nCOURSE EVIDENCE:\n{evidence}"
    try:
        with st.spinner("Generating locally..."):
            # Bypass system HTTP proxies for the local Ollama endpoint.
            local_http = requests.Session()
            local_http.trust_env = False
            with local_http:
                res = local_http.post(f"{OLLAMA}/api/generate", json={"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": -1}, timeout=300)
            res.raise_for_status()
            answer = res.json()["response"]
        st.subheader("Draft answer")
        st.markdown(answer)
        st.subheader("Retrieved evidence")
        for i, j in enumerate(ids):
            with st.expander(f"[S{i+1}] {records[j]['source']} · relevance {scores[j]:.2f}"):
                st.write(records[j]["text"])
    except Exception as e:
        st.error(f"Could not reach local Ollama model at {OLLAMA}: {e}")
