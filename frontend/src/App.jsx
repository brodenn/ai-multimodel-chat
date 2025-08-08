import React, { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import ModelSelector from "./components/ModelSelector";

export default function App() {
  const [selectedModel, setSelectedModel] = useState("deepseek-r1-distill");

  return (
    <div className="h-screen flex flex-col bg-gray-900">
      <header className="bg-gray-800 p-4 flex justify-between items-center">
        <h1 className="text-lg font-bold">AI Multi-Model Chat</h1>
        <ModelSelector selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
      </header>
      <ChatWindow model={selectedModel} />
    </div>
  );
}
