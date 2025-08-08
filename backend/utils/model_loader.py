import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "models_config.json"

with open(CONFIG_PATH) as f:
    MODELS_CONFIG = json.load(f)

loaded_models = {}
tokenizers = {}
active_model_name = None

def load_model(name: str, to_vram=True):
    global active_model_name

    if name not in MODELS_CONFIG:
        raise ValueError(f"Modellen {name} finns inte i config.json")

    if name in loaded_models:
        active_model_name = name
        return loaded_models[name], tokenizers[name]

    path = MODELS_CONFIG[name]["path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[LOADER] Laddar modell: {name} från {path} till {device}")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="auto" if to_vram else None
    )
    model.eval()

    loaded_models[name] = model
    tokenizers[name] = tokenizer
    active_model_name = name

    return model, tokenizer

def get_active_model():
    if not active_model_name:
        raise RuntimeError("Ingen modell är aktiv just nu.")
    return loaded_models[active_model_name], tokenizers[active_model_name], active_model_name
