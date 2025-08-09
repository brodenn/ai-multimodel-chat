from __future__ import annotations

import threading
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer

from utils.model_loader import load_model, get_active_model, list_models
from utils.prompt_utils import format_prompt
from utils.router import pick_model_by_prompt
from utils.rag import build_index, search as rag_search

# -------- App & CORS --------
app = FastAPI(title="AI Multi-Model Chat", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # lås ner till din frontend senare
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# -------- Startkonfig --------
# Ladda standardmodell (ligger kvar i RAM och växlas till VRAM vid generering)
load_model("deepseek-r1-distill")

# -------- Scheman --------
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    history: list | None = None
    model_name: str | None = None   # valfri override

class ModelSwitchRequest(BaseModel):
    model_name: str

class AskFileRequest(BaseModel):
    file_id: str
    question: str
    max_tokens: int = 200
    temperature: float = 0.3
    top_p: float = 0.9
    history: list | None = None
    model_name: str | None = None

# -------- Hjälp: läsa text ur filer --------
def read_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader
        chunks = []
        reader = PdfReader(str(path))
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(chunks)

    if ext == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Filformat stöds inte ännu: {ext}")

# -------- Hjälp: generera via streamer --------
def _stream_generate(model, tokenizer, inputs, gen_args):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_args = dict(gen_args, streamer=streamer)

    def _run():
        with torch.inference_mode():
            model.generate(**gen_args)

    threading.Thread(target=_run, daemon=True).start()
    return streamer

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Meta endpoints --------
@app.get("/health")
def health():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "device": dev}

@app.get("/active_model")
def active_model():
    _, _, name = get_active_model()
    return {"active_model": name}

@app.get("/models")
def models():
    return {"available": list_models()}

# -------- Modellhantering --------
@app.post("/switch_model")
def switch_model(req: ModelSwitchRequest):
    try:
        load_model(req.model_name)
        return {"status": "ok", "active_model": req.model_name}
    except Exception as e:
        raise HTTPException(400, f"Misslyckades byta modell: {e}")

@app.post("/route")
def route_preview(prompt: str):
    """Returnera vilken modell heuristiken skulle välja."""
    return {"suggested_model": pick_model_by_prompt(prompt)}

# -------- Textgenerering --------
@app.post("/generate")
def generate(req: PromptRequest):
    # Modellval: explicit > heuristik > aktiv
    chosen = (req.model_name or pick_model_by_prompt(req.prompt)).strip()
    load_model(chosen)

    model, tokenizer, model_name = get_active_model()
    device = _device()

    full_prompt = format_prompt(req.prompt, req.history)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    gen_args = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    streamer = _stream_generate(model, tokenizer, inputs, gen_args)
    return StreamingResponse(streamer, media_type="text/plain")

# -------- Filuppladdning + indexering (RAG) --------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Endast filer av typerna: {', '.join(sorted(allowed))}")

    file_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    try:
        with open(out_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(500, f"Kunde inte spara fil: {e}")

    # Läs och indexera direkt (FAISS)
    try:
        text = read_text_from_file(out_path)
        if not text.strip():
            raise ValueError("Tomt dokument.")
        build_index(file_id, text)
    except Exception as e:
        raise HTTPException(500, f"Kunde inte indexera filen: {e}")

    return {"file_id": file_id, "filename": file.filename}

# -------- Fråga på fil (RAG + LLM) --------
@app.post("/ask_file")
def ask_file(req: AskFileRequest):
    # Välj modell
    chosen = (req.model_name or pick_model_by_prompt(req.question)).strip()
    load_model(chosen)

    # Hämta relevanta bitar via FAISS
    try:
        hits = rag_search(req.file_id, req.question, k=5)
    except FileNotFoundError:
        raise HTTPException(404, "Index saknas. Ladda upp filen igen.")
    except Exception:
        raise HTTPException(500, "Kunde inte söka i indexet.")

    context = "\n\n".join([f"[{h['rank']+1}] {h['text']}" for h in hits])

    composed = (
        "Du är en hjälpsam assistent.\n"
        "Svara koncist och citera gärna [nummer] på de textbitar du använt.\n\n"
        "### Relevanta utdrag:\n"
        f"{context}\n\n"
        "### Fråga:\n"
        f"{req.question}\n\n"
        "### Svar:\n"
    )

    model, tokenizer, _ = get_active_model()
    device = _device()
    inputs = tokenizer(composed, return_tensors="pt").to(device)

    gen_args = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    streamer = _stream_generate(model, tokenizer, inputs, gen_args)
    return StreamingResponse(streamer, media_type="text/plain")

# -------- Fångare --------
@app.exception_handler(Exception)
async def unhandled_error(_, exc: Exception):
    # snyggare fallback-svar i JSON
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )
