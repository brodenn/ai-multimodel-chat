import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path

# Sökväg till models_config.json i projektroten
CONFIG_PATH = Path(__file__).parent.parent / "models_config.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    MODELS_CONFIG = json.load(f)

# Cache för laddade modeller och tokenizers
loaded_models = {}
tokenizers = {}
active_model_name = None


def list_models():
    """
    Returnerar en lista över alla modellnamn i models_config.json.
    """
    return list(MODELS_CONFIG.keys())


def load_model(name: str, to_vram: bool = True):
    """
    Ladda och aktivera en modell från models_config.json.
    Om modellen redan är laddad används cachen.
    """
    global active_model_name

    if name not in MODELS_CONFIG:
        raise ValueError(f"Modellen {name} finns inte i models_config.json")

    # Återanvänd redan laddad modell
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
    """
    Returnerar (modell, tokenizer, modellnamn) för nuvarande aktiva modellen.
    """
    if not active_model_name:
        raise RuntimeError("Ingen modell är aktiv just nu.")
    return loaded_models[active_model_name], tokenizers[active_model_name], active_model_name
