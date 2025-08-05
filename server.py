from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading

app = FastAPI()

# Modell och tokenizer
model_path = "../deepseek-r1-qwen32b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

class PromptRequest(BaseModel):
    prompt: str = "Vad är meningen med livet?"
    max_tokens: int = 200

@app.post("/generate")
def generate(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    # Skapa streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": req.max_tokens,
        "do_sample": True,  # gör att den inte försöker vara deterministisk
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return StreamingResponse(streamer, media_type="text/plain")
