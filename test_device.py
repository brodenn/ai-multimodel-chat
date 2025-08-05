from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "../deepseek-r1-qwen32b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Kontrollera var modellen är placerad
for name, param in model.named_parameters():
    print(f"{name} → {param.device}")
    break  # Vi kollar bara första parametern
