from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading

app = FastAPI()

# Ladda modellen
model_path = "../deepseek-r1-qwen32b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

@app.post("/generate")
def generate(req: PromptRequest):
    # Tokenisera och flytta till rätt enhet
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    # Skapa en streamer som fångar tokens live
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Starta genereringen i bakgrunden
    generation_kwargs = {
        **inputs,
        "max_new_tokens": req.max_tokens,
        "streamer": streamer
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Returnera stream till klienten
    return StreamingResponse(streamer, media_type="text/plain")
