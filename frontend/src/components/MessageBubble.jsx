import React from "react";

export default function MessageBubble({ role, text }) {
  const isUser = role === "User";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`p-3 rounded-lg max-w-lg ${
          isUser ? "bg-blue-500 text-white" : "bg-gray-700 text-white"
        }`}
      >
        {text}
      </div>
    </div>
  );
}
