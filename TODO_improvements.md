# Project Improvements TODO

## 🔍 SEO — indicizzazione su Google

- [ ] **Ottimizzare il `head` di `index.html`**: aggiungere meta description, meta author, Open Graph e structured data JSON-LD (schema.org `Person`).
- [ ] **Creare `sitemap.xml`** nella root con `https://riccardofigliozzi.github.io/`.
- [ ] **Creare `robots.txt`** nella root con `Allow: /` e riferimento alla sitemap.
- [ ] **Pushare le modifiche** (`git add index.html sitemap.xml robots.txt && git commit -m "Add SEO meta, sitemap and robots" && git push origin main`).
- [ ] **Registrare il sito su Google Search Console**: proprietà tipo "Prefisso URL", verificare con meta tag.
- [ ] **Inviare la sitemap e richiedere l'indicizzazione** della homepage.
- [ ] **Costruire autorevolezza**: link dal profilo LinkedIn e GitHub, menzione del nome completo nel sito, aggiornamenti regolari.

## 🤖 Chatbot con LangGraph (solo su Riccardo)

**Obiettivo**: chatbot flottante sul sito che risponde SOLO su Riccardo Figliozzi, nel suo stile e tono, usando LangGraph + RAG + LLM gratuito. La key del modello sta in un Cloudflare Worker (mai nel client).

### Architettura

```
[Browser — GitHub Pages]
   index.html + js/chat-widget.js   → widget flottante in basso a destra
                                          │  fetch POST /chat  (streaming SSE)
                                          ▼
[Cloudflare Worker — gratis]
   src/index.js  → LangGraph (langgraph.js) + RAG + Workers AI
      ├─ kb.json                    (fatti su di te: bio, CV, stile/tono)
      ├─ retrieve   → embedding bge-small (Workers AI) → cosine top-k
      ├─ relevance  → guardia: se la domanda NON riguarda Riccardo → risposta di cortesia
      └─ generate   → Qwen2.5-7B / Llama 3.1 8B con prompt "solo su Riccardo"
   wrangler.toml    (binding Workers AI)
```

### Passi

- [x] **Creare la cartella `worker/`** con `package.json`, `wrangler.toml`, `src/index.js`, `src/kb.json`.
- [x] **Creare `src/kb.json`**: knowledge base estratta da `index.html` + `cv/cv.pdf` + sezione "stile di scrittura" (prima persona, conciso, professionale ma informale). Chunking per argomento. Compilare i buchi (formazione dettagliata, progetti, interessi).
- [x] **Implementare il grafo LangGraph** in `src/index.js` (langgraph.js, gira nel Worker) con 3 nodi ed edge condizionali:
      - `retrieve`: embed domanda con `@cf/baai/bge-small-en-v1.5` → cosine similarity sui chunk → top-k.
      - `relevance`: guardia — se la domanda esula da Riccardo, risposta predefinita ("Mi occupo solo di Riccardo").
      - `generate`: system prompt = fatti recuperati + guida stile → risposta in streaming (SSE).
- [x] **Configurare `wrangler.toml`**: binding `AI` per Workers AI, impostare i CORS per `https://riccardofigliozzi.github.io/`.
- [x] **Modello LLM**: `@cf/qwen/qwen2.5-7b-instruct` (consigliato) oppure `@cf/meta/llama-3.1-8b-instruct`. Variante: chiamare la HuggingFace Inference API dal Worker con il token HF invece di Workers AI.
- [x] **Testare localmente**: `npx wrangler dev` (serve account Cloudflare: `npx wrangler login`). Unit test mock incluso: `npm test` in `worker/`.
- [x] **Creare il widget chat** in `js/chat-widget.js`: pallino flottante in basso a destra, finestra chat stile coerente col sito (colori da `css/style.css`), streaming delle risposte.
- [x] **Integrare il widget** in `index.html` (markup + script) e aggiungere gli stili a `css/style.css`.
- [x] **Deploy del Worker**: `npx wrangler deploy` → https://riccardo-chatbot.riccardofigliozzi.workers.dev, endpoint aggiornato nel widget.
- [x] **Push delle modifiche al sito** su GitHub Pages.
- [ ] **Check finale**: test end-to-end, controllo limite neuroni Workers AI (10k/giorno free), verifica CORS.

