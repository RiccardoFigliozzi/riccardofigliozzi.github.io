(function () {
    const ENDPOINT = window.RICCARDO_CHAT_ENDPOINT;
    if (!ENDPOINT) {
        console.warn("RICCARDO_CHAT_ENDPOINT not configured; chat widget disabled.");
        return;
    }

    const HISTORY_LIMIT = 8;
    let history = [];

    const root = document.createElement("div");
    root.className = "chat-widget";
    root.innerHTML = `
        <button class="chat-toggle" type="button" aria-label="Open chat">
            <svg class="chat-icon-open" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
            </svg>
            <svg class="chat-icon-close" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </button>
        <div class="chat-window" hidden>
            <header class="chat-header">
                <span class="chat-avatar">R</span>
                <div>
                    <strong>Riccardo's AI assistant</strong>
                    <small>Ask me anything about Riccardo</small>
                </div>
            </header>
            <div class="chat-messages" role="log" aria-live="polite"></div>
            <form class="chat-form">
                <input type="text" class="chat-input" placeholder="Type a message..." autocomplete="off" />
                <button type="submit" class="chat-send" aria-label="Send message">
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
    const form = root.querySelector(".chat-form");
    const input = root.querySelector(".chat-input");

    let isStreaming = false;

    toggle.addEventListener("click", () => {
        const open = windowEl.hidden;
        windowEl.hidden = !open;
        root.classList.toggle("open", open);
        if (open) {
            input.focus();
            if (!messagesEl.querySelector(".chat-msg")) {
                addMessage("bot", "Hi! I'm Riccardo's AI assistant. Ask me anything about Riccardo Figliozzi, his work, skills or how to contact him.");
            }
        }
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const text = input.value.trim();
        if (!text || isStreaming) return;
        input.value = "";
        send(text);
    });

    function addMessage(role, text) {
        const el = document.createElement("div");
        el.className = `chat-msg chat-msg-${role}`;
        el.innerHTML = `<span class="chat-bubble">${escapeHtml(text)}</span>`;
        messagesEl.appendChild(el);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return el.querySelector(".chat-bubble");
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

        const bubble = addMessage("bot", "…");
        isStreaming = true;
        form.querySelector(".chat-send").disabled = true;

        let answer = "";
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
                            answer += payload.token;
                            bubble.innerHTML = escapeHtml(answer);
                            messagesEl.scrollTop = messagesEl.scrollHeight;
                        }
                    }
                }
            }

            if (!answer) throw new Error("Empty response from server.");
        } catch (err) {
            bubble.innerHTML = escapeHtml("Sorry, something went wrong. Please try again later.");
            console.error("Chat error:", err);
        } finally {
            history.push({ role: "assistant", content: answer || "" });
            if (history.length > HISTORY_LIMIT) history = history.slice(-HISTORY_LIMIT);
            isStreaming = false;
            form.querySelector(".chat-send").disabled = false;
            input.focus();
        }
    }
})();
