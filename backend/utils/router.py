import re

def pick_model_by_prompt(prompt: str) -> str:
    """Välj modell baserat på prompt-innehåll (enkel heuristik).
    - Kodrelaterat → qwen3-coder-30b-a3b
    - Tungt resonemang/matte/logik → qwen3-30b-a3b-2507
    - Annars → deepseek-r1-distill
    """
    p = prompt.lower()

    code_hits = [
        r"\b(code|koda|funktion|bug|kompil|stack trace|regex|sql|javascript|python|c\+\+|java|rust|go|bash|shell)\b",
        r"```", r"class ", r"def ", r"public ", r"#include", r"SELECT .* FROM", r"\.py\b", r"\.js\b"
    ]
    reasoning_hits = [
        r"\b(bevisa|resonera|logik|steg för steg|matte|derivera|integral|ekvation|bevis|chain of thought)\b",
        r"\b(why|reason|prove|derivative|integral|equation|theorem)\b"
    ]

    if any(re.search(rx, p) for rx in code_hits):
        return "qwen3-coder-30b-a3b"

    if any(re.search(rx, p) for rx in reasoning_hits) or len(p) > 600:
        return "qwen3-30b-a3b-2507"

    return "deepseek-r1-distill"
