import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import kb from "./kb.json" with { type: "json" };

const EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5";
const LLM_MODEL = "@cf/meta/llama-3.1-8b-instruct-fp8";
const TOP_K = 4;
const RELEVANCE_THRESHOLD = 0.28;

const GREETINGS = {
  it: ["ciao", "salve", "buongiorno", "buonasera", "buona sera"],
  en: ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
  fr: ["bonjour", "bonsoir", "salut", "coucou"],
  es: ["hola", "buenos dias", "buenas tardes", "buenas noches"],
};

const OFF_TOPIC_MESSAGES = {
  it: "Sono Guidobaldo, l'assistente AI di Riccardo, quindi posso aiutarti solo con domande su Riccardo Figliozzi — il suo lavoro, le sue competenze, i suoi servizi o come contattarlo. Cosa ti piacerebbe sapere?",
  en: "I'm Guidobaldo, Riccardo's AI assistant, so I can only help with questions about Riccardo Figliozzi — his work, skills, services or how to contact him. What would you like to know?",
  fr: "Je suis Guidobaldo, l'assistant IA de Riccardo, je ne peux donc vous aider qu'avec des questions sur Riccardo Figliozzi — son travail, ses compétences, ses services ou comment le contacter. Que souhaitez-vous savoir ?",
  es: "Soy Guidobaldo, el asistente de IA de Riccardo, así que solo puedo ayudarte con preguntas sobre Riccardo Figliozzi: su trabajo, sus habilidades, sus servicios o cómo contactarlo. ¿Qué te gustaría saber?",
};
const GREETING_MESSAGES = {
  it: "Ciao! Sono Guidobaldo, l'assistente AI di Riccardo. Chiedimi tutto su Riccardo Figliozzi — la sua esperienza, le sue competenze, i suoi servizi o come contattarlo.",
  en: "Hey there! I'm Guidobaldo, Riccardo's AI assistant. Ask me anything about Riccardo Figliozzi — his experience, skills, services or how to get in touch.",
  fr: "Bonjour ! Je suis Guidobaldo, l'assistant IA de Riccardo. Posez-moi n'importe quelle question sur Riccardo Figliozzi — son expérience, ses compétences, ses services ou comment le contacter.",
  es: "¡Hola! Soy Guidobaldo, el asistente de IA de Riccardo. Pregúntame lo que quieras sobre Riccardo Figliozzi: su experiencia, sus habilidades, sus servicios o cómo contactarlo.",
};

let kbEmbeddingsCache = null;

function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function embedText(env, text) {
  const res = await env.AI.run(EMBEDDING_MODEL, { text: [text] });
  return res.data[0];
}

async function getKbEmbeddings(env) {
  if (kbEmbeddingsCache) return kbEmbeddingsCache;
  const res = await env.AI.run(EMBEDDING_MODEL, {
    text: kb.chunks.map((c) => c.content),
  });
  kbEmbeddingsCache = kb.chunks.map((chunk, i) => ({
    ...chunk,
    embedding: res.data[i],
  }));
  return kbEmbeddingsCache;
}

function detectGreeting(question) {
  const q = question.trim().toLowerCase().replace(/[?!.,]/g, "");
  for (const [lang, list] of Object.entries(GREETINGS)) {
    if (list.some((g) => q === g || q.startsWith(`${g} `))) return lang;
  }
  return null;
}

const LANGUAGE_HINTS = {
  it: [
    "il", "lo", "la", "gli", "i", "le", "sono", "quale", "quali", "come", "posso",
    "cosa", "perche", "mi", "ti", "tuo", "tua", "tuoi", "tue", "questo", "questa",
    "che", "un", "una", "della", "dei", "delle", "servizi", "competenze",
    "parlami", "contattarti", "consulenza", "lavoro", "esperienza", "chi",
  ],
  en: [
    "the", "and", "is", "are", "how", "what", "you", "your", "my", "with", "about",
    "tell", "me", "can", "of", "to", "in", "for", "do", "does", "did", "am",
    "services", "skills", "contact", "experience", "work", "who",
  ],
  fr: [
    "je", "tu", "vous", "comment", "quel", "quelle", "quels", "quelles", "pour",
    "avec", "est", "sont", "mon", "ma", "mes", "ton", "ta", "tes", "des", "les",
    "aux", "nous", "votre", "vos", "peux", "services", "competences", "contacter",
    "experience", "parle", "qui",
  ],
  es: [
    "hola", "como", "que", "cual", "cuales", "tu", "tus", "mi", "mis", "su", "sus",
    "para", "con", "es", "son", "eres", "estoy", "soy", "puedes", "quiero", "por",
    "servicios", "habilidades", "contactar", "experiencia", "quien",
  ],
};

function detectLanguage(question) {
  const tokens = question
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
  const counts = { it: 0, en: 0, fr: 0, es: 0 };
  for (const token of tokens) {
    for (const [lang, hints] of Object.entries(LANGUAGE_HINTS)) {
      if (hints.includes(token)) counts[lang] += 1;
    }
  }
  let best = "it";
  let bestCount = 0;
  for (const [lang, count] of Object.entries(counts)) {
    if (count > bestCount) {
      best = lang;
      bestCount = count;
    }
  }
  return best;
}

function buildSystemPrompt(question) {
  return `You are Guidobaldo, the virtual assistant of Riccardo Figliozzi, an AI Trainer and AI Consultant based in Florence, Italy.

CONTEXT ABOUT RICCARDO (use this as your only source of facts):
${kb.persona.style_guide.map((s) => `- ${s}`).join("\n")}

STRICT RULES:
- Answer ONLY about Riccardo Figliozzi. If the question is off-topic, politely refuse and redirect to Riccardo.
- Introduce yourself as Guidobaldo when relevant.
- Base your answer ONLY on the retrieved knowledge chunks below. Do not invent facts.
- Reply in Italian by default. Only switch to another language (English, French or Spanish) if the user writes in that language.
- Be concise: max 100 words. Use bullets only when helpful.
- If the chunks don't contain the answer, say you're not sure and suggest emailing riccardo.figliozzi@gmail.com.
- At the end of every relevant answer you may remind the user they can contact Riccardo at riccardo.figliozzi@gmail.com, but only if natural.
- Never mention that you have "chunks" or "a knowledge base".
`;
}

const GraphState = Annotation.Root({
  question: Annotation(),
  history: Annotation(),
  chunks: Annotation(),
  relevant: Annotation(),
  answer: Annotation(),
});

async function retrieveNode(state, config) {
  const { env } = config.configurable;
  const [questionEmbedding] = await Promise.all([
    embedText(env, state.question),
    getKbEmbeddings(env),
  ]);
  const scored = kbEmbeddingsCache
    .map((chunk) => ({ ...chunk, score: cosine(questionEmbedding, chunk.embedding) }))
    .sort((a, b) => b.score - a.score);
  const top = scored.slice(0, TOP_K).map(({ embedding, ...rest }) => rest);
  return { chunks: top };
}

async function relevanceNode(state) {
  const maxScore = state.chunks[0]?.score ?? 0;
  const relevant = maxScore >= RELEVANCE_THRESHOLD;
  return { relevant };
}

function routeRelevance(state) {
  return state.relevant ? "generate" : "fallback";
}

async function generateNode(state, config) {
  const { env, stream } = config.configurable;
  const context = state.chunks.map((c) => c.content).join("\n\n");
  const system = `${buildSystemPrompt(state.question)}\n\nRETRIEVED KNOWLEDGE:\n${context}`;
  const messages = [
    { role: "system", content: system },
    ...(state.history || []).slice(-6),
    { role: "user", content: state.question },
  ];

  let fullAnswer = "";
  const aiStream = await env.AI.run(LLM_MODEL, { messages, stream: true });
  const reader = aiStream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const event = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const line = event.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const payloadStr = line.slice(6);
        if (payloadStr === "[DONE]") break;
        let token = "";
        try {
          const parsed = JSON.parse(payloadStr);
          token = parsed.response ?? parsed.choices?.[0]?.delta?.content ?? "";
        } catch {
          token = payloadStr;
        }
        if (token) {
          fullAnswer += token;
          await stream.write(token);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  return { answer: fullAnswer };
}

async function fallbackNode(state, config) {
  const { stream } = config.configurable;
  const greetingLang = detectGreeting(state.question);
  const lang = greetingLang || detectLanguage(state.question);
  const message = greetingLang ? GREETING_MESSAGES[lang] : OFF_TOPIC_MESSAGES[lang];
  await stream.write(message);
  return { answer: message };
}

const graph = new StateGraph(GraphState)
  .addNode("retrieve", retrieveNode)
  .addNode("relevance", relevanceNode)
  .addNode("generate", generateNode)
  .addNode("fallback", fallbackNode)
  .addEdge(START, "retrieve")
  .addEdge("retrieve", "relevance")
  .addConditionalEdges("relevance", routeRelevance, ["generate", "fallback"])
  .addEdge("generate", END)
  .addEdge("fallback", END)
  .compile();

function corsHeaders(origin) {
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

function isAllowedOrigin(origin, env) {
  if (!origin) return false;
  const allowed = (env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim());
  return allowed.includes(origin);
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const cors = corsHeaders(isAllowedOrigin(origin, env) ? origin : null);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }

    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return Response.json({ ok: true }, { headers: cors });
    }

    if (request.method !== "POST" || url.pathname !== "/chat") {
      return Response.json({ error: "Not found" }, { status: 404, headers: cors });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return Response.json({ error: "Invalid JSON" }, { status: 400, headers: cors });
    }

    const question = String(body.message || "").trim();
    if (!question) {
      return Response.json({ error: "Empty message" }, { status: 400, headers: cors });
    }

    const encoder = new TextEncoder();
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();

    const stream = {
      write(text) {
        return writer.write(encoder.encode(`data: ${JSON.stringify({ token: text })}\n\n`));
      },
      close() {
        return writer.close();
      },
    };

    (async () => {
      try {
        await graph.invoke(
          {
            question,
            history: Array.isArray(body.history) ? body.history : [],
          },
          { configurable: { env, stream } }
        );
        await stream.close();
      } catch (err) {
        try {
          await stream.write(`\n\n[error: ${err.message}]`);
          await stream.close();
        } catch {
          // stream already closed
        }
      }
    })();

    return new Response(readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        ...cors,
      },
    });
  },
};
