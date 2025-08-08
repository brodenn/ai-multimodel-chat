def format_prompt(user_message: str, history=None):
    """
    Lägger till kontext och tidigare meddelanden.
    """
    if history is None:
        history = []
    prompt = ""
    for role, msg in history:
        prompt += f"{role}: {msg}\n"
    prompt += f"User: {user_message}\nAI:"
    return prompt
