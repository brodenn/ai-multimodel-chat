import React from "react";
import { switchModel } from "../services/api";

export default function ModelSelector({ selectedModel, setSelectedModel }) {
  const models = [
    { id: "deepseek-r1-distill", label: "DeepSeek R1 Distill" },
    { id: "qwen3-coder-30b-a3b", label: "Qwen3 Coder 30B" },
    { id: "qwen3-30b-a3b-2507", label: "Qwen3 30B 2507" },
    { id: "gpt-oss-20b", label: "GPT-OSS 20B" }
  ];

  const handleChange = async (e) => {
    const modelId = e.target.value;
    setSelectedModel(modelId);
    await switchModel(modelId);
  };

  return (
    <select
      value={selectedModel}
      onChange={handleChange}
      className="bg-gray-700 text-white p-2 rounded"
    >
      {models.map(m => (
        <option key={m.id} value={m.id}>{m.label}</option>
      ))}
    </select>
  );
}
