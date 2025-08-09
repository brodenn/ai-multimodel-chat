import React, { useState } from "react";
import MessageBubble from "./MessageBubble";
import { sendMessage, uploadFile, askFile } from "../services/api";

export default function ChatWindow({ model }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const [fileInfo, setFileInfo] = useState(null); // { file_id, filename }
  const [questionMode, setQuestionMode] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [sending, setSending] = useState(false);

  const historyPairs = () => messages.map((m) => [m.role, m.text]);

  const handleSend = async () => {
    if (!input.trim() || sending) return;

    const userMsg = { role: "User", text: input };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);

    const currentInput = input;
    setInput("");
    setSending(true);

    try {
      let reply = "";
      if (questionMode && fileInfo?.file_id) {
        reply = await askFile(fileInfo.file_id, currentInput, historyPairs(), model);
      } else {
        reply = await sendMessage(currentInput, historyPairs(), model);
      }
      setMessages((prev) => [...prev, { role: "AI", text: reply }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "AI", text: `⚠️ Fel: ${e.message || "Okänt fel"}` },
      ]);
    } finally {
      setSending(false);
    }
  };

  const handleUpload = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setUploadBusy(true);
    try {
      const info = await uploadFile(f);
      setFileInfo(info);
    } catch (e) {
      alert(e.message || "Uppladdning misslyckades");
    } finally {
      setUploadBusy(false);
    }
  };

  return (
    <div className="flex flex-col flex-1">
      {/* Toolbar */}
      <div className="p-3 bg-gray-800 flex items-center gap-3">
        <label className="bg-gray-700 px-3 py-2 rounded cursor-pointer">
          {uploadBusy ? "Laddar..." : "Ladda upp fil"}
          <input type="file" className="hidden" onChange={handleUpload} />
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={questionMode}
            onChange={(e) => setQuestionMode(e.target.checked)}
          />
          <span>Fråga om uppladdad fil</span>
        </label>
        {fileInfo && (
          <span className="text-sm text-gray-300">
            Aktiv fil: <b>{fileInfo.filename}</b>
          </span>
        )}
        <span className="ml-auto text-xs text-gray-400">
          Modell: <b>{model}</b>
        </span>
      </div>

      {/* Chat content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {messages.map((msg, idx) => (
          <MessageBubble key={idx} role={msg.role} text={msg.text} />
        ))}
      </div>

      {/* Input */}
      <div className="p-4 bg-gray-800 flex">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          className="flex-1 p-2 rounded bg-gray-700 text-white"
          placeholder={
            questionMode
              ? "Ställ en fråga om dokumentet..."
              : "Skriv ett meddelande..."
          }
          disabled={sending}
        />
        <button
          onClick={handleSend}
          className="ml-2 bg-blue-500 px-4 py-2 rounded text-white disabled:opacity-60"
          disabled={sending}
        >
          {sending ? "Skickar..." : "Skicka"}
        </button>
      </div>
    </div>
  );
}
