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
  if (!res.body) throw new Error("Saknar response body (stream).");

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

/**
 * Skicka ett vanligt chat-meddelande till /generate, med valfri historik.
 * Viktigt: skicka alltid model_name så frontendets val används per request.
 */
export async function sendMessage(prompt, history = [], model) {
  const controller = new AbortController();

  const res = await fetch(`${API}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Backend förväntar { prompt, history, model_name }
    body: JSON.stringify({ prompt, history, model_name: model }),
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

/**
 * Byt aktiv modell globalt i backend (valfritt om du skickar model_name per request).
 */
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

/**
 * Ladda upp en fil till /upload. Backend svarar med { file_id, filename }.
 */
export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/upload`, { method: "POST", body: form }).catch((e) => {
    throw new Error(`Uppladdning misslyckades: ${e.message}`);
  });
  return ensureOk(res).json(); // { file_id, filename }
}

/**
 * Ställ en fråga om en tidigare uppladdad fil via /ask_file.
 * Skicka med model_name för att styra modellen per request.
 */
export async function askFile(file_id, question, history = [], model) {
  const res = await fetch(`${API}/ask_file`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Backend förväntar { file_id, question, history, model_name }
    body: JSON.stringify({ file_id, question, history, model_name: model }),
  }).catch((e) => {
    throw new Error(`Kunde inte fråga på fil: ${e.message}`);
  });
  return streamToText(res);
}

// (Valfritt) Små hjälpare om du vill visa status i UI
export async function getHealth() {
  const res = await fetch(`${API}/health`).catch((e) => {
    throw new Error(`Health-check misslyckades: ${e.message}`);
  });
  return ensureOk(res).json();
}

export async function getModels() {
  const res = await fetch(`${API}/models`).catch((e) => {
    throw new Error(`Hämtning av modeller misslyckades: ${e.message}`);
  });
  return ensureOk(res).json();
}

export async function getActiveModel() {
  const res = await fetch(`${API}/active_model`).catch((e) => {
    throw new Error(`Hämtning av aktiv modell misslyckades: ${e.message}`);
  });
  return ensureOk(res).json();
}

// Exportera bas-URL om du vill logga/debugga
export const API_BASE = API;
