// frontend/src/services/api.js

// Bygg API-bas dynamiskt så det funkar när du öppnar sidan via VM:ens IP
// Ex: http://192.168.10.158:8000
const API_HOST = window.location.hostname || "localhost";
const API_PORT = 8000;
const API_PROTO = window.location.protocol.startsWith("https") ? "https" : "http";
const API = `${API_PROTO}://${API_HOST}:${API_PORT}`;

// Helpers
function ensureOk(res) {
  if (!res.ok) {
    throw new Error(`Server error ${res.status} ${res.statusText}`);
  }
  return res;
}

// Streama text (server-sent plain text)
export async function streamToText(res, signal) {
  ensureOk(res);
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let output = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    output += decoder.decode(value, { stream: true });
  }
  // flush sista chunken
  output += decoder.decode();
  return output;
}

export async function sendMessage(prompt, history = [], model) {
  const controller = new AbortController();
  const res = await fetch(`${API}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, history, model }),
    signal: controller.signal,
  }).catch((e) => {
    // Vanligt i browsern: adblock eller fel host → net::ERR_BLOCKED_BY_CLIENT / TypeError: Failed to fetch
    throw new Error(`Kunde inte nå API (${API}). Kontrollera IP/CORS/Adblock.\n${e.message}`);
  });

  try {
    return await streamToText(res, controller.signal);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new Error(`Fel vid läsning av stream: ${e.message}`);
  }
}

export async function switchModel(modelName) {
  const res = await fetch(`${API}/switch_model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName }),
  }).catch((e) => {
    throw new Error(`Kunde inte byta modell: ${e.message}`);
  });
  return ensureOk(res).json();
}

export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/upload`, { method: "POST", body: form }).catch((e) => {
    throw new Error(`Uppladdning misslyckades: ${e.message}`);
  });
  return ensureOk(res).json(); // { file_id, filename }
}

export async function askFile(file_id, question, history = [], model) {
  const res = await fetch(`${API}/ask_file`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id, question, history, model }),
  }).catch((e) => {
    throw new Error(`Kunde inte fråga på fil: ${e.message}`);
  });
  return streamToText(res);
}

// Exportera bas-URL om du vill logga/debugga
export const API_BASE = API;
