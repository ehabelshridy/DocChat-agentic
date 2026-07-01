// Same-origin now: FastAPI serves this file directly, so requests to
// "/health" and "/chat" automatically go to wherever this page is
// loaded from (e.g. http://127.0.0.1:8000). No separate frontend
// server or CORS setup needed.
const API_BASE_URL = "";
const MAX_RELEVANCE_RETRIES = 2;
const MAX_GENERATION_RETRIES = 2;

const chatArea = document.getElementById("chatArea");
const emptyState = document.getElementById("emptyState");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const connectionDot = document.getElementById("connectionDot");
const connectionLabel = document.getElementById("connectionLabel");

// ---------- Backend health check ----------

async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
    if (res.ok) {
      connectionDot.classList.add("online");
      connectionDot.classList.remove("offline");
      connectionLabel.textContent = "Backend connected";
    } else {
      throw new Error("Health check failed");
    }
  } catch (err) {
    connectionDot.classList.add("offline");
    connectionDot.classList.remove("online");
    connectionLabel.textContent = "Backend unreachable";
  }
}

checkBackendHealth();
setInterval(checkBackendHealth, 15000);

// ---------- Auto-growing textarea ----------

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

// ---------- Message rendering ----------

function hideEmptyState() {
  if (emptyState && emptyState.parentNode) {
    emptyState.remove();
  }
}

function appendUserMessage(text) {
  hideEmptyState();
  const wrapper = document.createElement("div");
  wrapper.className = "message user";
  wrapper.innerHTML = `
    <span class="message-label">You</span>
    <div class="message-bubble"></div>
  `;
  wrapper.querySelector(".message-bubble").textContent = text;
  chatArea.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function appendPendingMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant pending";
  wrapper.innerHTML = `
    <span class="message-label">DocChat</span>
    <div class="message-bubble">
      <span class="pulse"></span>
      <span>Retrieving and verifying…</span>
    </div>
  `;
  chatArea.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function statusItemHTML(ok, warnAtRetries, retryCount, maxRetries, label, okLabelSuffix) {
  // ok: boolean for final state; if not ok, decide warn vs bad based on retries used
  let cls = "ok";
  let text = `${label} ${okLabelSuffix}`;
  if (!ok) {
    cls = retryCount > 0 && retryCount <= maxRetries ? "warn" : "bad";
  }
  return `<span class="status-item ${cls}"><span class="dot"></span>${text}</span>`;
}

function renderReportCard({ isFallback, relevanceRetries, generationRetries }) {
  const relevantOk = !isFallback || relevanceRetries < MAX_RELEVANCE_RETRIES;
  const groundedOk = !isFallback;

  const relevanceStatus = isFallback && relevanceRetries >= MAX_RELEVANCE_RETRIES
    ? `<span class="status-item bad"><span class="dot"></span>RELEVANT ✕ (after ${relevanceRetries} retries)</span>`
    : `<span class="status-item ok"><span class="dot"></span>RELEVANT ✓${relevanceRetries > 0 ? ` (${relevanceRetries} retry)` : ""}</span>`;

  const groundedStatus = isFallback && relevanceRetries < MAX_RELEVANCE_RETRIES
    ? `<span class="status-item bad"><span class="dot"></span>GROUNDED ✕ (after ${generationRetries} retries)</span>`
    : !isFallback
      ? `<span class="status-item ok"><span class="dot"></span>GROUNDED ✓${generationRetries > 0 ? ` (${generationRetries} retry)` : ""}</span>`
      : "";

  return `${relevanceStatus}${groundedStatus}`;
}

function renderSourcesHTML(sources) {
  if (!sources || sources.length === 0) {
    return `<div class="sources-empty">No verified sources attached</div>`;
  }
  const tags = sources
    .map((s) => `<span class="source-tag">${escapeHTML(s.source_file)} · ${escapeHTML(s.section || "—")}</span>`)
    .join("");
  return `<div class="sources">${tags}</div>`;
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function replacePendingWithResult(pendingEl, data) {
  const isFallback = !data.sources || data.sources.length === 0;

  pendingEl.classList.remove("pending");
  if (isFallback) {
    pendingEl.classList.add("fallback");
  }

  const statusHTML = renderReportCard({
    isFallback,
    relevanceRetries: data.relevance_retries,
    generationRetries: data.generation_retries,
  });

  pendingEl.innerHTML = `
    <span class="message-label">DocChat</span>
    <div class="message-bubble"></div>
    <div class="report-card">
      <div class="status-strip">${statusHTML}</div>
      ${renderSourcesHTML(data.sources)}
    </div>
  `;
  pendingEl.querySelector(".message-bubble").textContent = data.answer;
  scrollToBottom();
}

function replacePendingWithError(pendingEl, message) {
  pendingEl.classList.remove("pending");
  pendingEl.classList.add("fallback");
  pendingEl.innerHTML = `
    <span class="message-label">DocChat</span>
    <div class="message-bubble"></div>
    <div class="report-card">
      <div class="status-strip">
        <span class="status-item bad"><span class="dot"></span>REQUEST FAILED</span>
      </div>
    </div>
  `;
  pendingEl.querySelector(".message-bubble").textContent = message;
  scrollToBottom();
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

// ---------- Submit handler ----------

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  appendUserMessage(question);
  questionInput.value = "";
  questionInput.style.height = "auto";

  sendButton.disabled = true;
  const pendingEl = appendPendingMessage();

  try {
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed with status ${res.status}`);
    }

    const data = await res.json();
    replacePendingWithResult(pendingEl, data);
  } catch (err) {
    replacePendingWithError(
      pendingEl,
      `Couldn't reach the backend: ${err.message}. Make sure the FastAPI server is running and serving this page from ${window.location.origin}.`
    );
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});
