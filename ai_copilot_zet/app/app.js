/**
 * app.js
 * ------
 * Zoho Desk widget logic for the AI Service Desk Copilot.
 *
 * Responsibilities:
 *  1. Initialize the Zoho Desk widget SDK and fetch the currently open ticket.
 *  2. Render a read-only ticket summary.
 *  3. On "Analyze Ticket" click, POST the ticket to the backend and render
 *     the structured AI response.
 *  4. Manage loading / empty / error UI states.
 *  5. Manage the Developer Mode toggle and diagnostics panel.
 *
 * No ticket data is ever written back to Zoho Desk from this file.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Point this at your running Flask backend. Override per-environment as needed.
// NOTE: port 5000 is used by `zet run` to serve the widget itself — the Flask
// backend must run on a different port (5001 by default; set PORT=5001 in
// backend/.env to match).
const API_BASE_URL = "http://localhost:5001";
const ANALYZE_ENDPOINT = `${API_BASE_URL}/api/analyze`;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentTicket = null;
let currentAnalysis = null;
let chatHistory = [];

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const el = {
  ticketSummary: document.getElementById("ticketSummary"),
  requesterHistoryCard: document.getElementById("requesterHistoryCard"),
  requesterHistoryBody: document.getElementById("requesterHistoryBody"),
  requesterHistoryHeader: document.getElementById("requesterHistoryHeader"),
  requesterHistoryCollapseBody: document.getElementById("requesterHistoryCollapseBody"),
  requesterHistoryChevron: document.getElementById("requesterHistoryChevron"),
  relatedTicketsCard: document.getElementById("relatedTicketsCard"),
  relatedTicketsBody: document.getElementById("relatedTicketsBody"),
  getRelatedBtn: document.getElementById("getRelatedBtn"),
  relatedTicketsLoading: document.getElementById("relatedTicketsLoading"),
  relatedTicketsStatusText: document.getElementById("relatedTicketsStatusText"),
  relatedTicketsHeader: document.getElementById("relatedTicketsHeader"),
  relatedTicketsCollapseBody: document.getElementById("relatedTicketsCollapseBody"),
  relatedTicketsChevron: document.getElementById("relatedTicketsChevron"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  loadingState: document.getElementById("loadingState"),
  loadingStatusText: document.getElementById("loadingStatusText"),
  errorState: document.getElementById("errorState"),
  analysisCard: document.getElementById("analysisCard"),
  chatCard: document.getElementById("chatCard"),
  chatMessages: document.getElementById("chatMessages"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  chatSendBtn: document.getElementById("chatSendBtn"),
  devModeToggle: document.getElementById("devModeToggle"),
  devPanel: document.getElementById("devPanel"),
  modelSelect: document.getElementById("modelSelect"),
  resetTicketBtn: document.getElementById("resetTicketBtn"),

  outSummary: document.getElementById("outSummary"),
  outCategory: document.getElementById("outCategory"),
  outSubCategory: document.getElementById("outSubCategory"),
  outPriority: document.getElementById("outPriority"),
  outConfidenceBar: document.getElementById("outConfidenceBar"),
  outConfidenceLabel: document.getElementById("outConfidenceLabel"),
  outCauses: document.getElementById("outCauses"),
  outSteps: document.getElementById("outSteps"),
  outCustomerReply: document.getElementById("outCustomerReply"),
  outTechNotes: document.getElementById("outTechNotes"),
  copyReplyBtn: document.getElementById("copyReplyBtn"),
  copyNotesBtn: document.getElementById("copyNotesBtn"),

  devModel: document.getElementById("devModel"),
  devProcessingTime: document.getElementById("devProcessingTime"),
  devTokens: document.getElementById("devTokens"),
  devValidation: document.getElementById("devValidation"),
  devTimestamp: document.getElementById("devTimestamp"),
  devWarnings: document.getElementById("devWarnings"),
  devPrompt: document.getElementById("devPrompt"),
  devRawResponse: document.getElementById("devRawResponse"),
};

// ---------------------------------------------------------------------------
// Per-ticket session cache
// ---------------------------------------------------------------------------
// Zoho reloads this widget's iframe from scratch every time the technician
// switches away from this subtab and back — that wipes all in-memory state
// (currentTicket, chatHistory, everything). sessionStorage, however, is
// scoped to the browser tab hosting Zoho Desk, not to this specific iframe
// instance, so it survives that reload. We key entries by ticket_id, so:
//   - same ticket, tab switch away and back -> restores exactly where the
//     technician left off (analysis, related tickets, chat).
//   - different ticket -> no matching key, so it just starts fresh there,
//     which is the "until I switch to a different ticket" behavior asked for.
// This is deliberately NOT Zoho's extension Data Storage feature (that's
// org-wide, rate-limited, and persists across sessions/devices — overkill
// for "don't lose my place when I flip tabs"). sessionStorage also clears
// itself when the browser tab closes, so nothing lingers indefinitely.
const CACHE_PREFIX = "copilot:ticket:";

function cacheKey(ticketId) {
  return `${CACHE_PREFIX}${ticketId}`;
}

function saveTicketCache(patch) {
  if (!currentTicket || !currentTicket.ticket_id) return;
  try {
    const existing = loadTicketCache(currentTicket.ticket_id) || {};
    const merged = Object.assign({}, existing, patch);
    sessionStorage.setItem(cacheKey(currentTicket.ticket_id), JSON.stringify(merged));
  } catch (err) {
    console.warn("Could not save session cache (non-fatal):", err);
  }
}

function loadTicketCache(ticketId) {
  try {
    const raw = sessionStorage.getItem(cacheKey(ticketId));
    return raw ? JSON.parse(raw) : null;
  } catch (err) {
    console.warn("Could not read session cache (non-fatal):", err);
    return null;
  }
}

/** Clears the cached entry for a single ticket only. */
function clearTicketCache(ticketId) {
  try {
    sessionStorage.removeItem(cacheKey(ticketId));
  } catch (err) {
    console.warn("Could not clear session cache (non-fatal):", err);
  }
}

/**
 * Re-render the UI from whatever was cached for this ticket, if anything.
 * Safe to call on every load — it's a no-op when there's no cache entry
 * (first time seeing this ticket this session).
 */
function restoreTicketCache(ticketId) {
  const cached = loadTicketCache(ticketId);
  if (!cached) return;

  if (cached.analysis) {
    renderAnalysis(cached.analysis);
    if (cached.developer) renderDevPanel(cached.developer);
    setUiState("result"); // note: this internally resets chat — restore it after.
    currentAnalysis = cached.analysis;
  }

  if (Array.isArray(cached.chatHistory) && cached.chatHistory.length > 0) {
    chatHistory = cached.chatHistory;
    chatHistory.forEach((msg) => appendChatBubble(msg.role, msg.text));
  }

  if (cached.relatedTickets) {
    renderRelatedTickets(cached.relatedTickets);
    el.relatedTicketsCollapseBody.classList.remove("d-none");
    el.relatedTicketsChevron.classList.add("expanded");
    el.relatedTicketsHeader.setAttribute("aria-expanded", "true");
    el.getRelatedBtn.innerHTML = '<i class="bi bi-arrow-clockwise me-1"></i> Refresh';
  }
}



document.addEventListener("DOMContentLoaded", () => {
  loadAvailableModels();
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();
});

function loadAvailableModels() {
  fetch(`${API_BASE_URL}/api/models`)
    .then((response) => response.json())
    .then((data) => {
      const models = data.available_models || [];
      const defaultModel = data.default_model;
      el.modelSelect.innerHTML = models
        .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`)
        .join("");
      if (defaultModel && models.includes(defaultModel)) {
        el.modelSelect.value = defaultModel;
      }
    })
    .catch((err) => {
      console.warn("Could not load model list from backend:", err);
      // Widget still works — omitting `model` in the analyze request just
      // means the backend uses its own configured default.
    });
}

window.onload = function () {
  if (typeof ZOHODESK === "undefined") {
    // Allows local testing outside of Zoho Desk (e.g. opening index.html directly).
    console.warn("ZOHODESK SDK not found — loading mock ticket for local testing.");
    loadMockTicket();
    return;
  }

  ZOHODESK.extension.onload().then(() => {
    // This content loads directly as a desk.ticket.detail.subtab widget —
    // it gets the full tab content pane automatically, no manual sizing
    // needed (no RESIZE call here; that was only relevant for the old
    // moreaction + modal setup).
    fetchCurrentTicket();
  });
};

function fetchCurrentTicket() {
  ZOHODESK.get("ticket")
    .then((data) => {
      const ticket = data && data.ticket ? data.ticket : data;
      currentTicket = normalizeTicket(ticket);
      renderTicketSummary(currentTicket);
      el.analyzeBtn.disabled = false;
      loadRequesterHistory(currentTicket);
      if (currentTicket.ticket_id) restoreTicketCache(currentTicket.ticket_id);
    })
    .catch((err) => {
      console.error("Failed to fetch ticket from Zoho Desk:", err);
      showError("Could not load the current ticket from Zoho Desk. Please reopen the widget.");
    });
}

/**
 * Fetch and render tickets related by Location + Sub-Category
 * (deterministic, identity-independent). Triggered manually by the "Get
 * Related Tickets" button rather than automatically on ticket load, since
 * the lookup can take a while at high ticket volume even parallelized —
 * better to make that an explicit, visible action than a silent background
 * fetch the technician might not realize is still running.
 */

/**
 * Wires up a click/keyboard toggle between a header and its collapsible
 * body — shared by Related Tickets and Requester History rather than
 * duplicating the same handler twice.
 */
function wireCollapsible(headerEl, bodyEl, chevronEl) {
  function toggle() {
    const isExpanded = !bodyEl.classList.contains("d-none");
    bodyEl.classList.toggle("d-none", isExpanded);
    chevronEl.classList.toggle("expanded", !isExpanded);
    headerEl.setAttribute("aria-expanded", String(!isExpanded));
  }
  headerEl.addEventListener("click", toggle);
  headerEl.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" || evt.key === " ") {
      evt.preventDefault();
      toggle();
    }
  });
}

wireCollapsible(el.relatedTicketsHeader, el.relatedTicketsCollapseBody, el.relatedTicketsChevron);
wireCollapsible(el.requesterHistoryHeader, el.requesterHistoryCollapseBody, el.requesterHistoryChevron);

el.getRelatedBtn.addEventListener("click", async () => {
  if (!currentTicket || !currentTicket.ticket_id) return;

  el.getRelatedBtn.disabled = true;
  el.getRelatedBtn.classList.add("d-none");
  el.relatedTicketsBody.innerHTML = "";
  el.relatedTicketsLoading.classList.remove("d-none");
  relatedTicketsRotator.start();

  try {
    const response = await fetch(
      `${API_BASE_URL}/api/ticket/${encodeURIComponent(currentTicket.ticket_id)}/related`
    );
    const data = await response.json();
    renderRelatedTickets(data);
    saveTicketCache({ relatedTickets: data });
    el.relatedTicketsBody.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    console.warn("Could not load related tickets:", err);
    el.relatedTicketsBody.innerHTML =
      '<span class="text-danger">Could not reach the AI Copilot backend. Please try again.</span>';
  } finally {
    relatedTicketsRotator.stop();
    el.relatedTicketsLoading.classList.add("d-none");
    el.getRelatedBtn.disabled = false;
    el.getRelatedBtn.classList.remove("d-none");
    el.getRelatedBtn.innerHTML = '<i class="bi bi-arrow-clockwise me-1"></i> Refresh';
  }
});

function renderRelatedTickets(data) {
  const related = data.related_tickets || [];

  if (!data.location || !data.sub_category) {
    // No Location/Sub-Category set on this ticket yet — nothing to match
    // against.
    el.relatedTicketsBody.innerHTML =
      '<span class="text-muted">This ticket has no Location or Sub-Category set, so related tickets can\'t be determined.</span>';
    currentTicket.related_tickets_summary = "";
    return;
  }

  if (related.length === 0) {
    el.relatedTicketsBody.innerHTML = `<span class="text-muted">No other "${escapeHtml(data.sub_category)}" tickets at this location recently.</span>`;
    currentTicket.related_tickets_summary = "";
    return;
  }

  const items = related
    .map(
      (t) =>
        `<li>${t.ticket_number ? `<span class="text-muted">#${escapeHtml(t.ticket_number)}</span> ` : ""}${escapeHtml(t.subject)} <span class="text-muted">(${escapeHtml(t.status)})</span></li>`
    )
    .join("");
  el.relatedTicketsBody.innerHTML = `
    <p class="mb-1">${related.length} related ticket${related.length === 1 ? "" : "s"} — same location, same sub-category:</p>
    <ul class="mb-0 ps-3">${items}</ul>
  `;

  currentTicket.related_tickets_summary = `${related.length} other "${data.sub_category}" tickets at the same location recently: ${related
    .map((t) => {
      const label = `${t.ticket_number ? `#${t.ticket_number} ` : ""}"${t.subject}" (${t.status})`;
      return t.resolution_snippet ? `${label} — recent agent activity: "${t.resolution_snippet}"` : label;
    })
    .join("; ")}.`;
}

/**
 * Fetch and render the requester's recent ticket history — a deterministic
 * lookup, no AI involved. Also builds a compact summary string attached to
 * currentTicket so the AI can reference it during /api/analyze without
 * fabricating history on its own.
 */
function loadRequesterHistory(ticket) {
  if (!ticket.requester || !ticket.ticket_id) {
    el.requesterHistoryCard.classList.add("d-none");
    return;
  }

  fetch(
    `${API_BASE_URL}/api/ticket/${encodeURIComponent(ticket.ticket_id)}/requester-history?email=${encodeURIComponent(ticket.requester)}`
  )
    .then((response) => response.json())
    .then((data) => {
      const tickets = data.tickets || [];
      el.requesterHistoryCard.classList.remove("d-none");

      if (tickets.length === 0) {
        el.requesterHistoryBody.innerHTML = '<span class="text-muted">No other recent tickets from this requester.</span>';
        currentTicket.requester_history_summary = "";
        return;
      }

      const items = tickets
        .map(
          (t) =>
            `<li>${t.ticket_number ? `<span class="text-muted">#${escapeHtml(t.ticket_number)}</span> ` : ""}${escapeHtml(t.subject)} <span class="text-muted">(${escapeHtml(t.status)})</span></li>`
        )
        .join("");
      el.requesterHistoryBody.innerHTML = `
        <p class="mb-1">${tickets.length} other recent ticket${tickets.length === 1 ? "" : "s"} from this requester:</p>
        <ul class="mb-0 ps-3">${items}</ul>
      `;

      currentTicket.requester_history_summary = `${tickets.length} other tickets in recent history: ${tickets
        .map((t) => `${t.ticket_number ? `#${t.ticket_number} ` : ""}"${t.subject}" (${t.status})`)
        .join("; ")}.`;
    })
    .catch((err) => {
      console.warn("Could not load requester history:", err);
      el.requesterHistoryCard.classList.add("d-none");
    });
}

/**
 * Fetch the latest thread's plain-text content for a ticket via our own
 * backend (which authenticates to Zoho's Desk API server-side). Using our
 * own backend — rather than ZOHODESK.request() directly from the widget —
 * avoids depending on the extension's "connectors" self-connection setup,
 * and reuses the same connect-src CSP entry already permitted for
 * /api/analyze. Returns a Promise resolving to a string (possibly empty).
 */
function fetchTicketConversation(ticketId) {
  return fetch(`${API_BASE_URL}/api/ticket/${encodeURIComponent(ticketId)}/description`)
    .then((response) => response.json())
    .then((body) => body.description || "");
}

/**
 * Normalize the ZOHODESK.get("ticket") response into the fields our backend
 * expects. Per Zoho's SDK docs, the raw ticket object uses flat fields like
 * `email` and `departmentId` (not nested objects) — adjust here if your
 * portal's field names differ (log `raw` to the console to check).
 */
function normalizeTicket(raw) {
  raw = raw || {};
  return {
    ticket_id: raw.ticketNumber || raw.id || raw.ticketId || "",
    subject: raw.subject || "",
    description: raw.description || raw.plainText || "",
    requester: raw.email || (raw.contact && raw.contact.email) || "",
    department: raw.departmentId || raw.departmentName || "",
    priority: raw.priority || "",
    status: raw.status || "",
    created_time: raw.createdTime || raw.createdTimeInSeconds || "",
  };
}

function loadMockTicket() {
  currentTicket = {
    ticket_id: "MOCK-1001",
    subject: "Slow Guest Network",
    description:
      "Team Reports the Guest Network is available but very slow",
    requester: "jane.doe@pendahealth.com",
    department: "IT Support",
    priority: "Medium",
    status: "Open",
    created_time: new Date().toISOString(),
  };
  renderTicketSummary(currentTicket);
  el.analyzeBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Rendering: Ticket Summary
// ---------------------------------------------------------------------------

function renderTicketSummary(ticket) {
  el.ticketSummary.innerHTML = `
    <dl class="row mb-0 small">
      <dt class="col-4">Subject</dt><dd class="col-8">${escapeHtml(ticket.subject)}</dd>
      <dt class="col-4">Requester</dt><dd class="col-8">${escapeHtml(ticket.requester)}</dd>
      <dt class="col-4">Created</dt><dd class="col-8">${escapeHtml(String(ticket.created_time))}</dd>
    </dl>
  `;
}

// ---------------------------------------------------------------------------
// Analyze button
// ---------------------------------------------------------------------------

el.analyzeBtn.addEventListener("click", async () => {
  if (!currentTicket) return;

  setUiState("loading");

  try {
    // Many tickets — especially those created via email — have a blank
    // top-level `description`; the actual message body lives in the
    // ticket's thread stream instead. Only fetch that enrichment now, at
    // analysis time, rather than on every ticket view — avoids an
    // unnecessary Zoho Desk API call for tickets the technician never
    // actually analyzes.
    if (!currentTicket.description && currentTicket.ticket_id) {
      try {
        const threadContent = await fetchTicketConversation(currentTicket.ticket_id);
        if (threadContent) {
          currentTicket.description = threadContent;
        }
      } catch (err) {
        console.warn("Could not fetch latest thread content:", err);
        // Non-fatal — proceed with analysis using whatever description
        // (possibly still blank) is already on the ticket.
      }
    }

    const response = await fetch(ANALYZE_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticket: currentTicket,
        developer_mode: el.devModeToggle.checked,
        model: el.modelSelect.value || undefined,
      }),
    });

    const body = await response.json().catch(() => ({}));

    if (!response.ok) {
      if (body.developer) renderDevPanel(body.developer);
      showError(body.error || "The AI analysis failed. Please try again.");
      return;
    }

    renderAnalysis(body.analysis);
    if (body.developer) renderDevPanel(body.developer);
    setUiState("result");
    currentAnalysis = body.analysis;
    saveTicketCache({ analysis: body.analysis, developer: body.developer || null });
  } catch (err) {
    console.error("Network error calling /api/analyze:", err);
    showError("Could not reach the AI Copilot backend. Check your connection and try again.");
  }
});

// ---------------------------------------------------------------------------
// Rendering: AI Analysis
// ---------------------------------------------------------------------------

function renderAnalysis(analysis) {
  analysis = analysis || {};

  el.outSummary.textContent = analysis.summary || "—";
  el.outCategory.textContent = analysis.category || "Unknown";
  el.outCategory.className = "copilot-chip copilot-chip-category";
  el.outSubCategory.textContent = analysis.sub_category || "—";
  el.outSubCategory.className = "copilot-chip copilot-chip-category";

  const priority = (analysis.suggested_priority || "").toLowerCase();
  el.outPriority.textContent = analysis.suggested_priority || "Unknown";
  el.outPriority.className = "copilot-chip " + priorityBadgeClass(priority);

  const confidence = clamp01(Number(analysis.confidence));
  el.outConfidenceBar.style.width = `${Math.round(confidence * 100)}%`;
  el.outConfidenceLabel.textContent = `${Math.round(confidence * 100)}% confidence`;

  renderList(el.outCauses, analysis.possible_causes);
  renderList(el.outSteps, analysis.troubleshooting_steps);

  el.outCustomerReply.textContent = analysis.customer_reply || "—";
  el.outTechNotes.textContent = analysis.technician_notes || "—";
}

function renderList(listEl, items) {
  listEl.innerHTML = "";
  if (!Array.isArray(items) || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "None identified.";
    li.className = "text-muted";
    listEl.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  });
}

function priorityBadgeClass(priority) {
  switch (priority) {
    case "low":
      return "priority-low";
    case "medium":
      return "priority-medium";
    case "high":
      return "priority-high";
    case "urgent":
      return "priority-urgent";
    default:
      return "copilot-chip-unknown";
  }
}

// ---------------------------------------------------------------------------
// Rendering: Developer Panel
// ---------------------------------------------------------------------------

function renderDevPanel(dev) {
  el.devModel.textContent = dev.model || "—";
  el.devProcessingTime.textContent = dev.processing_time_seconds != null ? `${dev.processing_time_seconds}s` : "—";
  el.devTokens.textContent = dev.estimated_tokens != null ? dev.estimated_tokens : "—";
  el.devValidation.textContent = dev.validation_status || "—";
  el.devTimestamp.textContent = dev.request_timestamp || "—";
  el.devPrompt.textContent = dev.prompt_sent || "";
  el.devRawResponse.textContent = dev.raw_response || "";

  el.devWarnings.innerHTML = "";
  if (Array.isArray(dev.warnings) && dev.warnings.length > 0) {
    const wrap = document.createElement("div");
    wrap.className = "alert alert-warning py-1 px-2 mb-2";
    wrap.innerHTML = dev.warnings.map((w) => `<div>${escapeHtml(w)}</div>`).join("");
    el.devWarnings.appendChild(wrap);
  }
}

el.devModeToggle.addEventListener("change", () => {
  el.devPanel.classList.toggle("d-none", !el.devModeToggle.checked);
});

el.resetTicketBtn.addEventListener("click", () => {
  if (!currentTicket || !currentTicket.ticket_id) return;

  const confirmed = window.confirm(
    "Reset this ticket? This clears its cached analysis, chat, and related tickets. Other tickets you've viewed this session are unaffected."
  );
  if (!confirmed) return;

  clearTicketCache(currentTicket.ticket_id);
  currentAnalysis = null;

  // Reset the current view back to its pre-analysis state. Ticket summary
  // and requester history stay as-is — those come straight from Zoho/the
  // backend on every load, not from the session cache.
  el.loadingState.classList.add("d-none");
  el.errorState.classList.add("d-none");
  el.analysisCard.classList.add("d-none");
  el.chatCard.classList.add("d-none");
  stopLoadingMessages();
  resetChat();

  el.relatedTicketsCollapseBody.classList.add("d-none");
  el.relatedTicketsChevron.classList.remove("expanded");
  el.relatedTicketsHeader.setAttribute("aria-expanded", "false");
  el.relatedTicketsBody.innerHTML = "";
  el.getRelatedBtn.innerHTML = '<i class="bi bi-search me-1"></i> Get Related Tickets';
  el.getRelatedBtn.classList.remove("d-none");
  el.getRelatedBtn.disabled = false;
});

// ---------------------------------------------------------------------------
// Rotating loading status messages (reusable — targets any element)
// ---------------------------------------------------------------------------

/**
 * Creates a rotator bound to a specific text element and message set.
 * Returns { start, stop } — start() begins cycling through `primary` in
 * order, falling into an indefinite loop through `extended` once
 * exhausted (never restarts from the beginning, which would look like the
 * process silently restarted rather than just taking longer).
 */
function createMessageRotator(targetEl, primary, extended, intervalMs = 1800) {
  let timer = null;
  return {
    start() {
      let index = 0;
      targetEl.textContent = primary[0];
      timer = setInterval(() => {
        index += 1;
        if (index < primary.length) {
          targetEl.textContent = primary[index];
        } else {
          const extendedIndex = (index - primary.length) % extended.length;
          targetEl.textContent = extended[extendedIndex];
        }
      }, intervalMs);
    },
    stop() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    },
  };
}

// Plays through once, in order, giving the impression of a real pipeline.
const analyzeRotator = createMessageRotator(
  el.loadingStatusText,
  [
    "Reading the ticket…",
    "Checking who's involved…",
    "Cross-referencing categories…",
    "Honing in on the issue…",
    "Weighing possible causes…",
    "Narrowing down priority…",
    "Sketching troubleshooting steps…",
    "Drafting a reply…",
    "Refining the tone…",
    "Double-checking the details…",
  ],
  ["Still working on it…", "This one's taking a bit longer…", "Almost there…", "Just finishing up…"]
);

const relatedTicketsRotator = createMessageRotator(
  el.relatedTicketsStatusText,
  [
    "Scanning recent tickets…",
    "Filtering by sub-category…",
    "Checking each ticket's location…",
    "Cross-referencing matches…",
  ],
  ["Still checking — this branch has a lot of tickets…", "Almost done…", "Wrapping up…"]
);

function startLoadingMessages() {
  analyzeRotator.start();
}

function stopLoadingMessages() {
  analyzeRotator.stop();
}

// ---------------------------------------------------------------------------
// UI state management
// ---------------------------------------------------------------------------

function setUiState(state) {
  el.loadingState.classList.add("d-none");
  el.errorState.classList.add("d-none");
  el.analysisCard.classList.add("d-none");
  stopLoadingMessages();

  if (state === "loading") {
    el.loadingState.classList.remove("d-none");
    el.chatCard.classList.add("d-none");
    startLoadingMessages();
    // Scrolls within .copilot-app's own overflow (see styles.css) — not
    // Zoho Desk's outer page — since html/body no longer scroll.
    el.loadingState.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } else if (state === "result") {
    el.analysisCard.classList.remove("d-none");
    el.chatCard.classList.remove("d-none");
    resetChat();
    el.analysisCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function showError(message) {
  stopLoadingMessages();
  el.errorState.textContent = message;
  el.errorState.classList.remove("d-none");
  el.loadingState.classList.add("d-none");
  el.errorState.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------------------
// Chat about this ticket
// ---------------------------------------------------------------------------

function resetChat() {
  chatHistory = [];
  el.chatMessages.innerHTML = "";
  el.chatInput.value = "";
}

function appendChatBubble(role, text, pending = false) {
  const bubble = document.createElement("div");
  bubble.className = `copilot-chat-bubble ${role}${pending ? " pending" : ""}`;
  bubble.textContent = text;
  el.chatMessages.appendChild(bubble);
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  return bubble;
}

el.chatForm.addEventListener("submit", async (evt) => {
  evt.preventDefault();
  const message = el.chatInput.value.trim();
  if (!message || !currentTicket) return;

  el.chatInput.value = "";
  el.chatInput.disabled = true;
  el.chatSendBtn.disabled = true;

  appendChatBubble("user", message);
  const pendingBubble = appendChatBubble("model", "Thinking…", true);

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticket: currentTicket,
        analysis: currentAnalysis || undefined,
        history: chatHistory,
        message: message,
        model: el.modelSelect.value || undefined,
      }),
    });
    const body = await response.json().catch(() => ({}));

    if (!response.ok) {
      pendingBubble.textContent = body.error || "Something went wrong. Please try again.";
      pendingBubble.classList.remove("pending");
      return;
    }

    pendingBubble.textContent = body.reply;
    pendingBubble.classList.remove("pending");
    chatHistory.push({ role: "user", text: message });
    chatHistory.push({ role: "model", text: body.reply });
    saveTicketCache({ chatHistory });
  } catch (err) {
    console.error("Chat request failed:", err);
    pendingBubble.textContent = "Could not reach the AI Copilot backend.";
    pendingBubble.classList.remove("pending");
  } finally {
    el.chatInput.disabled = false;
    el.chatSendBtn.disabled = false;
    el.chatInput.focus();
  }
});

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function clamp01(value) {
  if (Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function wireCopyButton(button, sourceEl) {
  button.addEventListener("click", () => {
    const text = sourceEl.textContent || "";
    navigator.clipboard
      .writeText(text)
      .then(() => {
        const originalIcon = button.innerHTML;
        button.innerHTML = '<i class="bi bi-check2"></i>';
        setTimeout(() => {
          button.innerHTML = originalIcon;
        }, 1500);
      })
      .catch((err) => console.error("Clipboard copy failed:", err));
  });
}

wireCopyButton(el.copyReplyBtn, el.outCustomerReply);
wireCopyButton(el.copyNotesBtn, el.outTechNotes);
