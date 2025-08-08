from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer
import threading, torch, uuid
from pathlib import Path

from utils.model_loader import load_model, get_active_model
from utils.prompt_utils import format_prompt
from utils.router import pick_model_by_prompt
from utils.rag import build_index, search as rag_search

from pypdf import PdfReader
from docx import Document

app = FastAPI(title="AI Multi-Model Chat")

ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Ladda standardmodell ---
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
    if ext in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(chunks)
    if ext in [".docx"]:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Filformat stöds inte ännu: {ext}")

# -------- Endpoints --------
@app.post("/switch_model")
def switch_model(req: ModelSwitchRequest):
    load_model(req.model_name)
    return {"status": "ok", "active_model": req.model_name}

@app.post("/route")
def route_preview(prompt: str):
    """Returnera vilken modell heuristiken skulle välja."""
    return {"suggested_model": pick_model_by_prompt(prompt)}

@app.post("/generate")
def generate(req: PromptRequest):
    # Modellval: explicit > heuristik > aktiv
    chosen = req.model_name or pick_model_by_prompt(req.prompt)
    load_model(chosen)

    model, tokenizer, model_name = get_active_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_prompt = format_prompt(req.prompt, req.history)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_args = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        use_cache=True,
    )

    def _run():
        with torch.inference_mode():
            model.generate(**gen_args)

    threading.Thread(target=_run, daemon=True).start()
    return StreamingResponse(streamer, media_type="text/plain")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Endast filer av typerna: {', '.join(sorted(allowed))}")

    file_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    with open(out_path, "wb") as f:
        f.write(await file.read())

    # Läs och indexera direkt (RAG)
    try:
        text = read_text_from_file(out_path)
        if not text.strip():
            raise ValueError("Tomt dokument.")
        build_index(file_id, text)
    except Exception as e:
        raise HTTPException(500, f"Kunde inte indexera filen: {e}")

    return {"file_id": file_id, "filename": file.filename}

@app.post("/ask_file")
def ask_file(req: AskFileRequest):
    # Välj modell
    chosen = req.model_name or pick_model_by_prompt(req.question)
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

    # Generera
    model, tokenizer, _ = get_active_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(composed, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_args = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        use_cache=True,
    )

    def _run():
        with torch.inference_mode():
            model.generate(**gen_args)

    threading.Thread(target=_run, daemon=True).start()
    return StreamingResponse(streamer, media_type="text/plain")
