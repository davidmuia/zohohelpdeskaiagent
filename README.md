# AI Service Desk Copilot (MVP)

An AI-powered assistant for technicians working Zoho Desk tickets. This MVP
proves a single capability: **read the currently open ticket, analyze it with
AI, and display the results in a clean widget.** Nothing is written back to
Zoho Desk — no ticket updates, no webhooks, no automation.

```
Read → Analyze → Display
```

---

## 1. Architecture Overview

```
service-desk-copilot/
├── backend/
│   ├── app.py            # Flask routes (thin — no AI or SQL logic here)
│   ├── ai_service.py      # Provider-agnostic AI abstraction (Gemini today)
│   ├── prompts.py         # All prompt text, isolated for fast iteration
│   ├── database.py        # SQLAlchemy engine/session setup
│   ├── models.py          # ticket_analysis ORM model
│   ├── config.py          # Env-driven configuration, single source of truth
│   ├── requirements.txt
│   └── .env.example
├── widget/
│   ├── index.html         # Zoho Desk widget UI (Bootstrap 5)
│   ├── app.js              # Widget logic: fetch ticket, call backend, render
│   ├── styles.css
│   └── plugin-manifest.json
└── README.md
```

**Key design decision — AI provider abstraction:** Flask routes never talk to
Gemini directly. They call `AIService.analyze_ticket()`, which delegates to
an `AIProvider` implementation (`GeminiProvider` today). Swapping in OpenAI,
Anthropic, Azure OpenAI, or a local model later means writing one new class
that implements `AIProvider` and registering it in `ai_service.py`'s
`_build_provider()` factory — no other code changes.

---

## 2. Obtaining a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey).
2. Sign in and click **Create API key**.
3. Copy the key — you'll paste it into `backend/.env` as `GEMINI_API_KEY`.

---

## 3. Running the Flask Backend Locally

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your-key-here

python app.py
```

The API will be available at `http://localhost:5000`. Confirm it's healthy:

```bash
curl http://localhost:5000/api/health
```

The SQLite database file (`service_desk_copilot.db`) is created automatically
on first run — no manual migration step needed for the MVP.

---

## 4. Installing the Zoho Desk Widget

Zoho Desk extensions are built and tested with **ZET (Zoho Extension
Toolkit)**, a CLI, not through a "create extension" menu inside Desk itself.
The high-level flow: scaffold a project with ZET → drop this repo's widget
files into it → run it locally against your Desk portal in Developer Mode →
pack and publish through **Zoho Sigma** when ready.

**a) Install prerequisites**

```bash
# Node.js 14+ and npm 8+ required — verify with:
node -v
npm -v

# Install the ZET CLI globally
npm install -g zoho-extension-toolkit   # Windows
sudo npm install -g zoho-extension-toolkit   # Mac/Linux
zet -v   # confirm install
```

**b) Scaffold a ZET project and bring in this widget's files**

```bash
zet init
# ? Select the Zoho service: Zoho Desk
# ? Project Name: ai-service-desk-copilot
# ? Need Module Support: No   (this repo doesn't need npm packages bundled)

cd ai-service-desk-copilot
```

ZET generates its own `app/` folder and a starter `plugin-manifest.json`.
Copy this repo's widget files into that generated `app/` folder, replacing
the starter files:

```bash
cp path/to/service-desk-copilot/widget/index.html   app/index.html
cp path/to/service-desk-copilot/widget/app.js       app/app.js
cp path/to/service-desk-copilot/widget/styles.css   app/styles.css
cp path/to/service-desk-copilot/widget/plugin-manifest.json  ./plugin-manifest.json
```

The `plugin-manifest.json` in this repo already sets `"location":
"desk.ticket.detail.rightpanel"` and points `"url"` at `/app/index.html`, so
the widget loads inside the ticket detail panel.

**c) Point the widget at your backend**

In `app/app.js`, set `API_BASE_URL` to your backend's **public** URL — Zoho
Desk is cloud-hosted and cannot reach `localhost`. For local development,
tunnel your Flask backend (e.g. `ngrok http 5000`) and use that URL.

**d) Run and test locally in Developer Mode**

```bash
zet run
```

This starts a local HTTPS server on port 5000 (install the SSL certificate
it generates at `cert.pem` once, per machine, to avoid browser "not secure"
warnings — ZET prints instructions the first time). Then, in Zoho Desk:

1. Click the **Setup** (gear) icon → **Developer Space → Build Extensions**.
2. Click **Enable Developer Mode**. The page refreshes and your locally
   running widget is injected live into the location set in
   `plugin-manifest.json` (the ticket right panel).
3. Open any ticket to see and test the widget. Edits to `app/index.html` /
   `app/app.js` show up on refresh — no repackaging needed while iterating.
4. Click **Disable Developer Mode** when done testing.

**e) Validate, pack, and publish via Zoho Sigma**

```bash
zet validate   # checks plugin-manifest.json against Desk's spec
zet pack       # produces a zip under ./dist/
```

1. Go to **Setup → Developer Space → Build Extensions → Sigma** (or
   sigma.zoho.com directly).
2. Click **New Extension**, name it, select **Zoho Desk** as the service,
   and upload the zip from `dist/`.
3. Save as draft, review the extension details, then **Publish**.
4. Click the generated **Install URL**, choose the departments/profiles that
   should see the widget, and click **Install**.

---

## 5. Local Testing (Without Zoho Desk)

`widget/app.js` detects when the `ZOHODESK` SDK object isn't present (e.g.
opening `index.html` directly in a browser) and loads a mock ticket instead,
so you can test the full UI and API flow before wiring up ZET at all:

```bash
cd widget
python3 -m http.server 8080
# open http://localhost:8080 in a browser, with the backend running on :5000
```

Because the widget and backend run on different origins during local
testing, make sure `CORS_ALLOWED_ORIGINS` in `.env` permits your widget's
origin (or leave it as `*` for local development only).

---

## 6. Packaging the Extension

Packaging is handled by the ZET CLI, not a manual zip step (see §4e):

```bash
zet validate
zet pack
```

The output zip lands in `dist/` inside your ZET project and is what you
upload to Zoho Sigma. Re-run `zet validate && zet pack` and re-upload
whenever the widget changes.

---

## 7. Developer Mode

Toggle **Dev Mode** in the widget header to reveal a diagnostics panel with:
the exact prompt sent, the raw AI response, processing time, model used,
estimated token usage, validation status, warnings, and request timestamp.

This is gated in two places:
- **Client:** the toggle simply requests diagnostics from the backend.
- **Server:** `DEVELOPER_MODE_ALLOWED` in `.env` is a hard kill switch. Set
  it to `false` in production and the backend will never include the
  `developer` payload, regardless of what the widget requests.

---

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Widget shows "Could not reach the AI Copilot backend" | `API_BASE_URL` unreachable from Zoho Desk's servers | Use a public URL (ngrok, deployed host); `localhost` won't work once hosted in Zoho |
| `/api/health` returns `ai_reachable: false` | Missing/invalid `GEMINI_API_KEY` | Check `.env`, regenerate key in AI Studio |
| "The AI analysis could not be completed" | Model returned non-JSON or malformed JSON | Check Dev Mode's raw response; adjust `prompts.py` if the model is drifting from the schema |
| CORS errors in browser console | Widget origin not allowed | Update `CORS_ALLOWED_ORIGINS` in `.env` |
| Empty ticket fields in summary | Zoho ticket payload shape differs from expected | Adjust `normalizeTicket()` in `app.js` to match your Desk instance's field names |
| SQLite "database is locked" | Concurrent writes under high load | For production scale, point `DATABASE_URL` at Postgres/MySQL instead |

---

## 9. Roadmap: From MVP to Full Copilot Platform

This MVP is the read-only foundation. Suggested evolution, roughly in order:

**Phase 2 — Automation-adjacent (still human-approved)**
- Zoho Desk webhooks to trigger analysis automatically on ticket creation
- Automatic categorization suggestions written back to the ticket (with
  technician confirmation, not silent auto-apply)
- Similar-ticket search (requires embeddings + a vector store)

**Phase 3 — Knowledge & context expansion**
- Knowledge Base article generation from resolved tickets
- RAG over the KB so `AIService` grounds answers in prior solutions
- Asset Management integration so the AI knows what hardware/software a
  requester has

**Phase 4 — Systems integration**
- Active Directory integration (account status, lockouts, group membership)
- Monitoring integrations (pull relevant alerts for the affected system)
- Network diagnostics (ping/traceroute-style checks surfaced to the AI)

**Phase 5 — Interactive & autonomous**
- AI Chat: a conversational interface layered on top of ticket context
- Multi-agent architecture: specialized agents (triage, diagnostics,
  drafting, escalation) coordinated by an orchestrator
- Guarded automation: AI-proposed actions (replies, status changes) that a
  technician approves with one click, moving toward supervised autonomy

Each phase builds on the same `AIProvider` abstraction and `ticket_analysis`
audit table established in this MVP, so earlier work is never thrown away —
only extended.
