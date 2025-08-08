const API = "http://localhost:8000";

export async function sendMessage(prompt, history = []) {
  const res = await fetch(`${API}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, history })
  });
  return streamToText(res);
}

export async function switchModel(modelName) {
  return fetch(`${API}/switch_model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName })
  }).then(res => res.json());
}

export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Uppladdning misslyckades");
  return res.json(); // { file_id, filename }
}

export async function askFile(file_id, question, history = []) {
  const res = await fetch(`${API}/ask_file`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id, question, history })
  });
  return streamToText(res);
}

async function streamToText(res) {
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
