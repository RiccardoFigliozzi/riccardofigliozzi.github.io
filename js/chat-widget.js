(function () {
    const ENDPOINT = window.RICCARDO_CHAT_ENDPOINT;
    if (!ENDPOINT) {
        console.warn("RICCARDO_CHAT_ENDPOINT not configured; chat widget disabled.");
        return;
    }

    const HISTORY_LIMIT = 8;
    const MAX_INPUT_CHARS = 2000;
    const GREETING = "Ciao a tutti! Sono Guidubaldo, l'assistente di Riccardo. Chiedimi tutto su Riccardo Figliozzi, il suo lavoro, le sue competenze o come contattarlo.";
    const SUGGESTIONS = [
        "Quali servizi offre Riccardo?",
        "Quali sono le competenze AI di Riccardo?",
        "Come posso contattare Riccardo?",
    ];

    let history = [];
    let started = false;

    const root = document.createElement("div");
    root.className = "chat-widget";
    root.innerHTML = `
        <button class="chat-toggle" type="button" aria-label="Apri chat">
            <svg class="chat-icon-open" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
            </svg>
            <svg class="chat-icon-close" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </button>
        <div class="chat-window" hidden>
            <header class="chat-header">
                <span class="chat-avatar">G</span>
                <div>
                    <strong>Guidubaldo</strong>
                    <small><span class="chat-status-dot"></span> Online &middot; risponde subito</small>
                </div>
                <button class="chat-close" type="button" aria-label="Chiudi chat">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </header>
            <div class="chat-messages" role="log" aria-live="polite"></div>
            <div class="chat-suggestions" hidden></div>
            <form class="chat-form">
                <input type="text" class="chat-input" placeholder="Chiedimi qualcosa..." autocomplete="off" />
                <button type="submit" class="chat-send" aria-label="Invia messaggio">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
                    </svg>
                </button>
            </form>
        </div>
    `;

    document.body.appendChild(root);

    const toggle = root.querySelector(".chat-toggle");
    const windowEl = root.querySelector(".chat-window");
    const messagesEl = root.querySelector(".chat-messages");
    const suggestionsEl = root.querySelector(".chat-suggestions");
    const closeBtn = root.querySelector(".chat-close");
    const form = root.querySelector(".chat-form");
    const input = root.querySelector(".chat-input");
    const sendBtn = form.querySelector(".chat-send");

    let isStreaming = false;

    SUGGESTIONS.forEach((text) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chat-chip";
        chip.textContent = text;
        chip.addEventListener("click", () => {
            input.value = text;
            form.dispatchEvent(new Event("submit", { cancelable: true }));
        });
        suggestionsEl.appendChild(chip);
    });

    function open() {
        windowEl.hidden = false;
        root.classList.add("open");
        toggle.setAttribute("aria-label", "Chiudi chat");
        if (!started) {
            addMessage("bot", GREETING);
            suggestionsEl.hidden = false;
            started = true;
        }
        input.focus();
    }

    function close() {
        windowEl.hidden = true;
        root.classList.remove("open");
        toggle.setAttribute("aria-label", "Apri chat");
    }

    toggle.addEventListener("click", () => {
        if (windowEl.hidden) open();
        else close();
    });

    closeBtn.addEventListener("click", close);

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !windowEl.hidden) close();
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const text = input.value.trim();
        if (!text || isStreaming) return;
        if (text.length > MAX_INPUT_CHARS) {
            input.value = "";
            suggestionsEl.hidden = true;
            addMessage("bot", "Il messaggio è troppo lungo: massimo " + MAX_INPUT_CHARS + " caratteri.");
            return;
        }
        input.value = "";
        suggestionsEl.hidden = true;
        send(text);
    });

    function scrollToBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function addMessage(role, text) {
        const el = document.createElement("div");
        el.className = `chat-msg chat-msg-${role}`;
        if (role === "bot") {
            el.innerHTML = `<span class="chat-msg-avatar">G</span><span class="chat-bubble">${escapeHtml(text)}</span>`;
        } else {
            el.innerHTML = `<span class="chat-bubble">${escapeHtml(text)}</span>`;
        }
        messagesEl.appendChild(el);
        scrollToBottom();
        return el.querySelector(".chat-bubble");
    }

    function addTyping() {
        const el = document.createElement("div");
        el.className = "chat-msg chat-msg-bot";
        el.innerHTML = `<span class="chat-msg-avatar">G</span><span class="chat-bubble chat-typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;
        messagesEl.appendChild(el);
        scrollToBottom();
        return el;
    }

    function escapeHtml(str) {
        return str.replace(/[&<>"']/g, (c) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
        }[c]));
    }

    async function send(text) {
        addMessage("user", text);
        history.push({ role: "user", content: text });
        if (history.length > HISTORY_LIMIT) history = history.slice(-HISTORY_LIMIT);

        const typing = addTyping();
        isStreaming = true;
        sendBtn.disabled = true;

        let answer = "";
        let bubble = null;
        try {
            const response = await fetch(ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, history }),
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || `Request failed (${response.status})`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                let idx;
                while ((idx = buffer.indexOf("\n\n")) >= 0) {
                    const event = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);
                    for (const line of event.split("\n")) {
                        if (!line.startsWith("data: ")) continue;
                        const payload = JSON.parse(line.slice(6));
                        if (payload.token) {
                            if (!bubble) {
                                typing.remove();
                                bubble = addMessage("bot", "");
                            }
                            answer += payload.token;
                            bubble.innerHTML = escapeHtml(answer);
                            scrollToBottom();
                        }
                    }
                }
            }

            if (!answer) throw new Error("Empty response from server.");
        } catch (err) {
            typing.remove();
            const friendly = {
                "Request limit reached": "Hai esaurito le 5 richieste disponibili. Ricarica la pagina o riprova più tardi.",
                "Message too long": "Il messaggio è troppo lungo: massimo 2000 caratteri.",
            }[err.message];
            addMessage("bot", friendly || "Ops, qualcosa è andato storto. Riprova più tardi.");
            console.error("Chat error:", err);
        } finally {
            history.push({ role: "assistant", content: answer || "" });
            if (history.length > HISTORY_LIMIT) history = history.slice(-HISTORY_LIMIT);
            isStreaming = false;
            sendBtn.disabled = false;
            input.focus();
        }
    }
})();
