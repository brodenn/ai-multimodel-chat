from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Multispråk (svenska m.fl.)
EMB_MODEL_NAME = "intfloat/multilingual-e5-base"
_model = None

INDEX_DIR = Path(__file__).parent.parent / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

def _get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL_NAME, trust_remote_code=True)
        _model.max_seq_length = 512
    return _model

def _chunk_text(text: str, chunk_size=900, overlap=150):
    text = text.replace("\r", "")
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def _paths_for_file(file_id: str):
    base = INDEX_DIR / file_id
    return {
        "meta": base.with_suffix(".json"),
        "faiss": base.with_suffix(".faiss"),
        "npy": base.with_suffix(".npy"),
    }

def build_index(file_id: str, full_text: str):
    """Skapa FAISS-index från fulltext."""
    em = _get_embedder()
    chunks = _chunk_text(full_text)
    if not chunks:
        raise ValueError("Inget extraherat innehåll för indexering.")

    # E5 kräver "query: ..." / "passage: ..." konvention
    passages = [f"passage: {c}" for c in chunks]
    emb = em.encode(passages, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    paths = _paths_for_file(file_id)
    faiss.write_index(index, str(paths["faiss"]))
    np.save(str(paths["npy"]), emb)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "embedding_model": EMB_MODEL_NAME}, f, ensure_ascii=False)

def search(file_id: str, query: str, k: int = 5):
    paths = _paths_for_file(file_id)
    if not all(p.exists() for p in paths.values()):
        raise FileNotFoundError("Index saknas för denna fil. Kör upload igen.")

    index = faiss.read_index(str(paths["faiss"]))
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunks = meta["chunks"]

    em = _get_embedder()
    q_vec = em.encode([f"query: {query}"], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype="float32")
    D, I = index.search(q_vec, k)
    hits = []
    for rank, idx in enumerate(I[0]):
        if 0 <= idx < len(chunks):
            hits.append({"rank": rank, "score": float(D[0][rank]), "text": chunks[idx]})
    return hits
