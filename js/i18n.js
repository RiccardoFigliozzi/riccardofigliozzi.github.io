(function () {
    const I18N = {
        it: {
            "nav.services": "Servizi",
            "nav.contact": "Contatti",
            "nav.cv": "CV",
            "about.role": "AI Trainer & Consulente",
            "about.description": "Educatore e consulente di <strong>Intelligenza Artificiale, AI Automation e Data Science</strong>, con esperienza specifica nell'implementazione di modelli LLM e agenti AI. <a href=\"#contact\" style=\"color: white; text-shadow: 2px 2px 0 #424242;\"> <strong>Contattami</strong></a>",
            "sketch.hint": "Clicca, trascina e lascia che accada la magia ;)",
            "services.automation.title": "AI Automation & Consulenza",
            "services.automation.body": "Analisi dei processi aziendali per identificare e implementare workflow di automazione basati su AI usando strumenti come <strong>n8n</strong>.",
            "services.generative.title": "AI Generativa",
            "services.generative.body": "Progettazione e sviluppo di <strong>chatbot agentici e soluzioni basate su LLM</strong> per l'automazione avanzata dei processi e il coinvolgimento degli utenti.",
            "services.data.title": "Data Science & Analytics",
            "services.data.body": "Sviluppo di soluzioni di <strong>Machine Learning</strong>, pipeline ETL e estrazione di insight utili da dati complessi.",
            "services.more": "Prenota una chiamata con me",
            "trust.experience": "Anni di Esperienza",
            "trust.professionals": "Professionisti Formati",
            "trust.approach": "Approccio Pratico",
            "trust.technologies": "n8n, LangChain, LLMs",
            "chat.ariaOpen": "Apri chat",
            "chat.ariaClose": "Chiudi chat",
            "chat.aiLabel": "Assistente AI",
            "chat.ariaAiLabel": "Questo chatbot è un'intelligenza artificiale",
            "chat.status": "Online &middot; risponde subito",
            "chat.placeholder": "Chiedimi qualcosa...",
            "chat.send": "Invia messaggio",
            "chat.greeting": "Ciao! Sono Guidubaldo, un assistente AI creato da Riccardo Figliozzi. Sono qui per raccontarti tutto su di lui: lavoro, competenze, servizi o come prenotare una call. Chiedimi quello che vuoi!",
            "chat.suggestion.1": "Quali servizi offre Riccardo?",
            "chat.suggestion.2": "Come posso prenotare una call con Riccardo?",
            "chat.suggestion.3": "Come posso contattare Riccardo?",
            "chat.error.tooLong": "Il messaggio è troppo lungo: massimo 2000 caratteri.",
            "chat.error.limit": "Hai esaurito le 5 richieste disponibili. Ricarica la pagina o riprova più tardi.",
            "chat.error.generic": "Ops, qualcosa è andato storto. Riprova più tardi.",
        },
        en: {
            "nav.services": "Services",
            "nav.contact": "Contact",
            "nav.cv": "CV",
            "about.role": "AI Trainer & Consultant",
            "about.description": "Educator and consultant of <strong>Artificial Intelligence, AI Automation and Data Science</strong>, with specific expertise in implementing LLM models and AI agents. <a href=\"#contact\" style=\"color: white; text-shadow: 2px 2px 0 #424242;\"> <strong>Contact me</strong></a>",
            "sketch.hint": "Click, Drag and let the magic happen ;)",
            "services.automation.title": "AI Automation & Consulting",
            "services.automation.body": "Analyzing business processes to identify and implement AI-driven automation workflows using tools like <strong>n8n</strong>.",
            "services.generative.title": "Generative AI",
            "services.generative.body": "Designing and developing <strong>agentic chatbots and LLM-based solutions</strong> for advanced process automation and user engagement.",
            "services.data.title": "Data Science & Analytics",
            "services.data.body": "Crafting <strong>Machine Learning</strong> solutions, building ETL pipelines, and extracting actionable insights from complex data.",
            "services.more": "Schedule a call with me",
            "trust.experience": "Years Experience",
            "trust.professionals": "Professionals Trained",
            "trust.approach": "Practical Approach",
            "trust.technologies": "n8n, LangChain, LLMs",
            "chat.ariaOpen": "Open chat",
            "chat.ariaClose": "Close chat",
            "chat.aiLabel": "AI Assistant",
            "chat.ariaAiLabel": "This chatbot is an artificial intelligence",
            "chat.status": "Online &middot; replies quickly",
            "chat.placeholder": "Ask me something...",
            "chat.send": "Send message",
            "chat.greeting": "Hi! I'm Guidubaldo, an AI assistant created by Riccardo Figliozzi. I'm here to tell you everything about him: his work, skills, services or how to book a call. Ask me anything!",
            "chat.suggestion.1": "What services does Riccardo offer?",
            "chat.suggestion.2": "How can I book a call with Riccardo?",
            "chat.suggestion.3": "How can I contact Riccardo?",
            "chat.error.tooLong": "Message too long: maximum 2000 characters.",
            "chat.error.limit": "You have used up your 5 available requests. Reload the page or try again later.",
            "chat.error.generic": "Oops, something went wrong. Please try again later.",
        },
    };

    const STORAGE_KEY = "riccardo-lang";
    const SUPPORTED = ["it", "en"];

    function detectDefault() {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (SUPPORTED.includes(stored)) return stored;
        const nav = (navigator.language || "it").toLowerCase();
        if (nav.startsWith("en")) return "en";
        return "it";
    }

    let currentLang = detectDefault();

    function apply(lang) {
        currentLang = lang;
        document.documentElement.lang = lang;
        const dict = I18N[lang];
        document.querySelectorAll("[data-i18n]").forEach((el) => {
            const key = el.getAttribute("data-i18n");
            if (dict[key] !== undefined) el.innerHTML = dict[key];
        });
        document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
            const key = el.getAttribute("data-i18n-placeholder");
            if (dict[key] !== undefined) el.placeholder = dict[key];
        });
        window.RICCARDO_CHAT_LANG = lang;
        document.querySelectorAll(".lang-toggle").forEach((btn) => {
            btn.classList.toggle("active", btn.getAttribute("data-lang") === lang);
        });
        document.dispatchEvent(new CustomEvent("riccardo:lang", { detail: lang }));
    }

    function setLang(lang) {
        if (!SUPPORTED.includes(lang)) return;
        localStorage.setItem(STORAGE_KEY, lang);
        apply(lang);
    }

    window.RICCARDO_I18N = {
        get current() {
            return currentLang;
        },
        t(key) {
            return I18N[currentLang][key] !== undefined ? I18N[currentLang][key] : I18N.en[key];
        },
        setLang,
        supported: SUPPORTED,
    };

    document.addEventListener("DOMContentLoaded", () => apply(currentLang));
})();
