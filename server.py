from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
import torch
import threading

app = FastAPI()

# Modellväg (justera om din ligger på annan plats)
model_path = "../deepseek-r1-qwen32b"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Välj datatyp baserat på GPU-stöd
if torch.cuda.is_available():
    dtype = torch.float16
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    dtype = torch.float16
    device = torch.device("mps")
else:
    dtype = torch.float32
    device = torch.device("cpu")

# Ladda modellen
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto" if device.type != "cpu" else None
)
model.eval()

# Pydantic-modell för förfrågan
class PromptRequest(BaseModel):
    prompt: str = "Vad är meningen med livet?"
    max_tokens: int = 200

# Genererings-API
@app.post("/generate")
def generate(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_args = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": req.max_tokens,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer
    }

    # Starta generering i separat tråd
    thread = threading.Thread(target=model.generate, kwargs=gen_args)
    thread.start()

    return StreamingResponse(streamer, media_type="text/plain")
