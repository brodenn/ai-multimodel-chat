from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
import torch
import threading

app = FastAPI()

# Modell och tokenizer
model_path = "../deepseek-r1-qwen32b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Konfiguration för 4-bit kvantisering (bitsandbytes)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # 🔧 RÄTT: använd torch.float16, inte "fp16"
)

# Ladda modellen med kvantisering och device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map="auto"
)

# Request-schema
class PromptRequest(BaseModel):
    prompt: str = "Vad är meningen med livet?"
    max_tokens: int = 200

# Genererings-API
@app.post("/generate")
def generate(req: PromptRequest):
    # Flytta input till samma device som modellen
    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    # Streamer för svar
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Parametrar för generering
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": req.max_tokens,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer
    }

    # Kör modellen i en separat tråd
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Skicka tillbaka strömmande svar
    return StreamingResponse(streamer, media_type="text/plain")
