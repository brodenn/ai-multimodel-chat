export async function sendMessage(prompt, history = [], model = null) {
  const res = await fetch("http://localhost:8000/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, history, model })
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let output = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    output += decoder.decode(value);
  }
  return output;
}

export async function switchModel(modelName) {
  return fetch("http://localhost:8000/switch_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName })
  }).then(res => res.json());
}
