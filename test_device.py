from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "../deepseek-r1-qwen32b"

# Ladda tokenizer
print("🔁 Laddar tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Ladda modell
print("🔁 Laddar modell...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Kontrollera vilken enhet modellen ligger på
device_info = {}
for name, param in model.named_parameters():
    print(f"✅ Första parametern: {name} → {param.device}")
    break  # Endast första parametern

# Extra: kontrollera om modellen alls använder GPU
gpu_detected = any(param.device.type == "cuda" for param in model.parameters())

if gpu_detected:
    print("🚀 Modellen är laddad på GPU! ✔️")
else:
    print("⚠️ Modellen ligger fortfarande på CPU. ❌")
    print("💡 Tips:")
    print(" - Kontrollera att ROCm fungerar med PyTorch (testa: torch.cuda.is_available())")
    print(" - Använd istället .to('cuda') manuellt")
    print(" - Undvik device_map='auto' om den ignoreras av ROCm")

# Bekräfta att CUDA är tillgängligt
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
