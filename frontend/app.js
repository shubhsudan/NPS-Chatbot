/* app.js — NPS Chatbot frontend logic */

// ---------------------------------------------------------------------------
// Config — update BACKEND_URL after deploying to Render
// ---------------------------------------------------------------------------
// Empty string = same-origin (works on Render). Override window.BACKEND_URL for cross-origin dev.
const BACKEND_URL = window.BACKEND_URL || "";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const chatWindow = document.getElementById("chatWindow");
const queryInput = document.getElementById("queryInput");
const sendBtn    = document.getElementById("sendBtn");

// ---------------------------------------------------------------------------
// Auto-resize textarea
// ---------------------------------------------------------------------------
queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";
});

// Send on Enter (Shift+Enter = newline)
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});
sendBtn.addEventListener("click", handleSend);

// ---------------------------------------------------------------------------
// Suggestion buttons
// ---------------------------------------------------------------------------
function fillSuggestion(btn) {
  queryInput.value = btn.textContent.trim();
  queryInput.focus();
  queryInput.dispatchEvent(new Event("input"));
}

// ---------------------------------------------------------------------------
// Message rendering helpers
// ---------------------------------------------------------------------------

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user";
  div.innerHTML = `
    <div class="message-avatar" aria-hidden="true">👤</div>
    <div class="message-body">
      <div class="message-text">${escapeHtml(text)}</div>
    </div>`;
  chatWindow.appendChild(div);
  scrollToBottom();
}

function appendTypingIndicator() {
  const div = document.createElement("div");
  div.className = "message bot typing-indicator";
  div.id = "typingIndicator";
  div.innerHTML = `
    <div class="message-avatar" aria-hidden="true">🤖</div>
    <div class="message-body">
      <div class="message-text">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>
    </div>`;
  chatWindow.appendChild(div);
  scrollToBottom();
}

function removeTypingIndicator() {
  document.getElementById("typingIndicator")?.remove();
}

function appendBotMessage(data, originalQuery) {
  const piiChanged = data.sanitized_query !== originalQuery;

  const citationsHtml = buildCitationsHtml(data.citations);
  const piiHtml = piiChanged
    ? `<div class="pii-notice">⚠️ Personal identifiers were removed from your question before processing.</div>`
    : "";

  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = `
    <div class="message-avatar" aria-hidden="true">🤖</div>
    <div class="message-body">
      <div class="message-text">${formatAnswer(data.answer)}</div>
      ${piiHtml}
      ${citationsHtml}
    </div>`;

  chatWindow.appendChild(div);

  // Wire up citations toggle
  const toggle = div.querySelector(".citations-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const list = div.querySelector(".citations-list");
      const isOpen = list.classList.toggle("visible");
      toggle.classList.toggle("open", isOpen);
      toggle.querySelector(".arrow").textContent = isOpen ? "▶" : "▶";
    });
  }

  scrollToBottom();
}

function appendErrorMessage(msg) {
  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = `
    <div class="message-avatar" aria-hidden="true">🤖</div>
    <div class="message-body">
      <div class="message-text error">⚠️ ${escapeHtml(msg)}</div>
    </div>`;
  chatWindow.appendChild(div);
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Citations HTML builder
// ---------------------------------------------------------------------------
function buildCitationsHtml(citations) {
  if (!citations || citations.length === 0) return "";

  const items = citations.map(c => `
    <div class="citation-item">
      <a href="${escapeHtml(c.source_url)}" target="_blank" rel="noopener noreferrer">
        ${escapeHtml(c.title)}
      </a>
      <span class="citation-badge">${escapeHtml(c.doc_type || "document")}</span>
    </div>`).join("");

  return `
    <div class="citations">
      <button class="citations-toggle" aria-expanded="false">
        <span class="arrow">▶</span>
        ${citations.length} source${citations.length > 1 ? "s" : ""}
      </button>
      <div class="citations-list" role="list">${items}</div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Answer formatting — convert newlines and basic markdown to HTML
// ---------------------------------------------------------------------------
function formatAnswer(text) {
  // Escape HTML first, then re-apply formatting
  let html = escapeHtml(text);

  // Bold: **text**
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Numbered lists: lines starting with "1. "
  html = html.replace(/((?:^\d+\.\s.+\n?)+)/gm, (block) => {
    const items = block.trim().split("\n").map(l =>
      `<li>${l.replace(/^\d+\.\s/, "").trim()}</li>`
    ).join("");
    return `<ol>${items}</ol>`;
  });

  // Bullet lists: lines starting with "- " or "• "
  html = html.replace(/((?:^[-•]\s.+\n?)+)/gm, (block) => {
    const items = block.trim().split("\n").map(l =>
      `<li>${l.replace(/^[-•]\s/, "").trim()}</li>`
    ).join("");
    return `<ul>${items}</ul>`;
  });

  // Paragraphs: double newlines
  html = html
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(Boolean)
    .map(p => (p.startsWith("<ol>") || p.startsWith("<ul>")) ? p : `<p>${p}</p>`)
    .join("");

  // Single newlines within paragraphs
  html = html.replace(/<p>(.*?)<\/p>/gs, (_, content) =>
    `<p>${content.replace(/\n/g, "<br>")}</p>`
  );

  return html;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setLoading(on) {
  sendBtn.disabled = on;
  queryInput.disabled = on;
}

// ---------------------------------------------------------------------------
// Main send handler
// ---------------------------------------------------------------------------
async function handleSend() {
  const query = queryInput.value.trim();
  if (!query) return;

  // Clear input
  queryInput.value = "";
  queryInput.style.height = "auto";

  // Remove welcome message on first real question
  document.getElementById("welcomeMsg")?.remove();

  appendUserMessage(query);
  appendTypingIndicator();
  setLoading(true);

  try {
    const resp = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    removeTypingIndicator();

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const detail = Array.isArray(err.detail)
        ? err.detail.map(e => e.msg).join(", ")
        : (err.detail || `Server error ${resp.status}`);
      throw new Error(resp.status === 429
        ? "⏳ The AI service is rate-limited right now. Please wait a minute and try again."
        : detail);
    }

    const data = await resp.json();
    appendBotMessage(data, query);

  } catch (err) {
    removeTypingIndicator();
    appendErrorMessage(
      err.message === "Failed to fetch"
        ? "Could not reach the server. Please check your connection or try again shortly."
        : err.message
    );
  } finally {
    setLoading(false);
    queryInput.focus();
  }
}
