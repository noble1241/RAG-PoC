import { useState, useRef } from "react";
import { streamChat, ingestText, ingestFile } from "./api";
import type { SourceChunk, IngestResponse } from "./api";
import "./App.css";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const [pasteText, setPasteText] = useState("");
  const [ingestSource, setIngestSource] = useState("paste");
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResponse | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function sendChat(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || chatLoading) return;
    const userMsg = query.trim();
    setQuery("");
    setChatError(null);

    const userMsgObj: Message = { role: "user", content: userMsg };
    const assistantMsgObj: Message = { role: "assistant", content: "", sources: [] };

    setMessages((m) => [...m, userMsgObj, assistantMsgObj]);
    const assistantIdx = messages.length + 1;
    setChatLoading(true);

    try {
      for await (const event of streamChat(userMsg)) {
        if (event.type === "sources") {
          setMessages((m) =>
            m.map((msg, i) => (i === assistantIdx ? { ...msg, sources: event.sources } : msg))
          );
        } else if (event.type === "token") {
          setMessages((m) =>
            m.map((msg, i) =>
              i === assistantIdx ? { ...msg, content: msg.content + (event.content ?? "") } : msg
            )
          );
          bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        }
      }
    } catch (err) {
      setChatError(String(err));
    } finally {
      setChatLoading(false);
    }
  }

  async function handleIngestText(e: React.FormEvent) {
    e.preventDefault();
    if (!pasteText.trim() || ingestLoading) return;
    setIngestLoading(true);
    setIngestError(null);
    setIngestResult(null);
    try {
      const result = await ingestText(pasteText.trim(), ingestSource || "paste");
      setIngestResult(result);
      setPasteText("");
    } catch (err) {
      setIngestError(String(err));
    } finally {
      setIngestLoading(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setIngestLoading(true);
    setIngestError(null);
    setIngestResult(null);
    try {
      const result = await ingestFile(file);
      setIngestResult(result);
    } catch (err) {
      setIngestError(String(err));
    } finally {
      setIngestLoading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>Ingest Documents</h2>
        <form onSubmit={handleIngestText} className="ingest-form">
          <input
            value={ingestSource}
            onChange={(e) => setIngestSource(e.target.value)}
            placeholder="Source name"
            className="input"
          />
          <textarea
            value={pasteText}
            onChange={(e) => setPasteText(e.target.value)}
            placeholder="Paste text here…"
            rows={8}
            className="textarea"
          />
          <button type="submit" disabled={ingestLoading || !pasteText.trim()} className="btn">
            {ingestLoading ? "Ingesting…" : "Ingest Text"}
          </button>
        </form>

        <div className="file-upload">
          <label className="btn btn-secondary">
            Upload File (.txt / .md / .pdf)
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.md,.pdf"
              onChange={handleFileUpload}
              hidden
            />
          </label>
        </div>

        {ingestLoading && <p className="status">Processing…</p>}
        {ingestError && <p className="error">{ingestError}</p>}
        {ingestResult && (
          <div className="result-card">
            <strong>{ingestResult.source}</strong> ingested
            <br />
            {ingestResult.chunk_count} chunks · {ingestResult.tokens_processed} tokens
          </div>
        )}
      </aside>

      <main className="chat">
        <h1>RAG Chat</h1>
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message message--${msg.role}`}>
              <div className="message__content">
                {msg.content ||
                  (msg.role === "assistant" && chatLoading && i === messages.length - 1
                    ? "▌"
                    : "")}
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <details className="sources">
                  <summary>Sources ({msg.sources.length})</summary>
                  {msg.sources.map((s, j) => (
                    <div key={j} className="source-chunk">
                      <strong>{s.source}</strong> chunk {s.chunk_index} · score {s.score}
                      <p>{s.content.slice(0, 200)}{s.content.length > 200 ? "…" : ""}</p>
                    </div>
                  ))}
                </details>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {chatError && <p className="error">{chatError}</p>}

        <form onSubmit={sendChat} className="chat-form">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question…"
            disabled={chatLoading}
            className="input"
          />
          <button type="submit" disabled={chatLoading || !query.trim()} className="btn">
            {chatLoading ? "…" : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}
