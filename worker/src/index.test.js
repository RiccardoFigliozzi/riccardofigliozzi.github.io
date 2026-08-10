import test from "node:test";
import assert from "node:assert/strict";
import kb from "./kb.json" with { type: "json" };

const EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5";
const LLM_MODEL = "@cf/meta/llama-3.1-8b-instruct-fp8";

const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with", "is", "are",
  "was", "were", "be", "been", "his", "her", "it", "as", "by", "from", "that", "this",
  "he", "she", "they", "them", "do", "does", "did", "how", "what", "where", "who", "why",
  "tell", "me", "about", "work", "works",
]);

function hashToEmbedding(str) {
  const vec = new Array(256).fill(0);
  const clean = str.toLowerCase().replace(/[^a-z0-9\s]/g, "");
  const tokens = clean.split(/\s+/).filter(Boolean);
  const keep = tokens.filter((t) => !STOPWORDS.has(t));
  const text = keep.join(" ");
  for (let i = 0; i < text.length - 2; i++) {
    const gram = text.slice(i, i + 3);
    let h = 2166136261;
    for (let j = 0; j < gram.length; j++) {
      h ^= gram.charCodeAt(j);
      h = Math.imul(h, 16777619);
    }
    vec[Math.abs(h) % vec.length] += 1;
  }
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  return vec.map((v) => v / norm);
}

const env = {
  ALLOWED_ORIGINS: "https://riccardofigliozzi.github.io,http://localhost:8000",
  AI: {
    async run(model, opts) {
      if (model === EMBEDDING_MODEL) {
        const data = opts.text.map((t) => hashToEmbedding(t));
        return { data };
      }
      if (model === LLM_MODEL) {
        const stream = new ReadableStream({
          start(controller) {
            const words = "Riccardo is an AI consultant in Florence.".split(" ");
            controller.enqueue(
              new TextEncoder().encode(
                words
                  .slice(0, 4)
                  .map((w) => `data: ${JSON.stringify({ response: w + " " })}\n\n`)
                  .join("")
              )
            );
            setTimeout(() => {
              controller.enqueue(
                new TextEncoder().encode(
                  words
                    .slice(4)
                    .map((w) => `data: ${JSON.stringify({ response: w + " " })}\n\n`)
                    .join("")
                )
              );
              controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
              controller.close();
            }, 1);
          },
        });
        return stream;
      }
      throw new Error(`unexpected model ${model}`);
    },
  },
};

async function collect(worker, envVars, message) {
  const response = await worker.fetch(
    new Request("https://worker/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "https://riccardofigliozzi.github.io" },
      body: JSON.stringify({ message }),
    }),
    envVars
  );
  assert.equal(response.status, 200);
  const reader = response.body.getReader();
  let buffer = "";
  let full = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += new TextDecoder().decode(value);
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const event = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = event.split("\n").find((l) => l.startsWith("data: "));
      if (line) {
        const payload = JSON.parse(line.slice(6));
        if (payload.token) full += payload.token;
      }
    }
  }
  return full;
}

test("kb.json has chunks and persona", () => {
  assert.ok(Array.isArray(kb.chunks));
  assert.ok(kb.chunks.length >= 15);
  assert.equal(kb.persona.name, "Riccardo Figliozzi");
});

test("relevance routes on-topic question to generate and streams LLM tokens", async () => {
  const mod = await import("./index.js");
  const { default: worker } = mod;
  const text = await collect(worker, env, "Riccardo works at Data Masters?");
  assert.match(text, /Riccardo is an AI consultant/);
});

test("relevance routes off-topic question to fallback", async () => {
  const mod = await import("./index.js");
  const { default: worker } = mod;
  const text = await collect(worker, env, "Tell me how to bake a sourdough bread.");
  assert.match(text, /only help with questions about Riccardo/i);
});

test("relevance routes greeting to fallback greeting message", async () => {
  const mod = await import("./index.js");
  const { default: worker } = mod;
  const text = await collect(worker, env, "Hello!");
  assert.match(text, /Guidobaldo, Riccardo's AI assistant/);
});

test("Italian greeting gets an Italian reply", async () => {
  const mod = await import("./index.js");
  const { default: worker } = mod;
  const text = await collect(worker, env, "Ciao!");
  assert.match(text, /Guidobaldo/);
});

test("off-topic question defaults to Italian", async () => {
  const mod = await import("./index.js");
  const { default: worker } = mod;
  const text = await collect(worker, env, "Manutenzione della bici da corsa");
  assert.match(text, /Sono Guidobaldo/);
});

test("English off-topic question gets an English reply", async () => {
  const mod = await import("./index.js");
  const { default: worker } = mod;
  const text = await collect(worker, env, "How do I cook pasta from scratch?");
  assert.match(text, /I'm Guidobaldo, Riccardo's AI assistant/);
});

test("CORS allows configured origin and blocks others", async () => {
  const mod = await import("./index.js");
  const { default: worker } = mod;

  const allowed = await worker.fetch(
    new Request("https://worker/health", {
      headers: { Origin: "https://riccardofigliozzi.github.io" },
    }),
    env
  );
  assert.equal(allowed.headers.get("Access-Control-Allow-Origin"), "https://riccardofigliozzi.github.io");

  const blocked = await worker.fetch(
    new Request("https://worker/health", { headers: { Origin: "https://evil.example.com" } }),
    env
  );
  assert.notEqual(blocked.headers.get("Access-Control-Allow-Origin"), "https://evil.example.com");
});

test("empty message returns 400", async () => {
  const mod = await import("./index.js");
  const { default: worker } = mod;
  const response = await worker.fetch(
    new Request("https://worker/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "https://riccardofigliozzi.github.io" },
      body: JSON.stringify({ message: "   " }),
    }),
    env
  );
  assert.equal(response.status, 400);
});
