const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

export interface SourceChunk {
  source: string;
  chunk_index: number;
  content: string;
  score: number;
}

export interface ChatEvent {
  type: "sources" | "token" | "done";
  sources?: SourceChunk[];
  content?: string;
}

export interface IngestResponse {
  document_id: string;
  source: string;
  chunk_count: number;
  tokens_processed: number;
}

export async function* streamChat(query: string): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(String(err.detail ?? res.statusText));
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.trim();
      if (line.startsWith("data: ")) {
        yield JSON.parse(line.slice(6)) as ChatEvent;
      }
    }
  }
}

export async function ingestText(text: string, source: string): Promise<IngestResponse> {
  const res = await fetch(`${BASE}/documents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(String(err.detail ?? res.statusText));
  }
  return res.json() as Promise<IngestResponse>;
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(String(err.detail ?? res.statusText));
  }
  return res.json() as Promise<IngestResponse>;
}
