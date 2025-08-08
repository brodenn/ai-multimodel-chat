from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import threading
import torch
from transformers import TextIteratorStreamer
from utils.model_loader import load_model, get_active_model
from utils.prompt_utils import format_prompt

app = FastAPI()

# Ladda första modellen direkt (DeepSeek)
load_model("deepseek-r1-distill")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    history: list = None

class ModelSwitchRequest(BaseModel):
    model_name: str

@app.post("/switch_model")
def switch_model(req: ModelSwitchRequest):
    load_model(req.model_name)
    return {"status": "ok", "active_model": req.model_name}

@app.post("/generate")
def generate(req: PromptRequest):
    model, tokenizer, model_name = get_active_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_prompt = format_prompt(req.prompt, req.history)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_args = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": req.max_tokens,
        "do_sample": True,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
        "use_cache": True
    }

    thread = threading.Thread(target=model.generate, kwargs=gen_args)
    thread.start()

    return StreamingResponse(streamer, media_type="text/plain")
