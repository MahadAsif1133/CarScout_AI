# CarScout_AI (formerly GaadiScout) — Autonomous AI Car Deal Finder

> Find underpriced cars on PakWheels.com automatically — using ML price prediction,
> a Groq or local Ollama AI evaluator, LangGraph orchestration, optional LangSmith tracing,
> optional MongoDB raw listing archiving, real-time n8n / Telegram / email alerts, and a live Gradio dashboard.
>
> *Note on project naming: The repository is **CarScout_AI**; internal filenames (`gaadiscout.db`) and the GitHub Actions workflow name (**GaadiScout Pipeline**) use the original **GaadiScout** name and refer to the exact same project.*

---

## 📸 Dashboard Preview

CarScout_AI's 4-tab Gradio dashboard gives you:
- **Live Deals** — sortable/filterable table of all current deals
- **Price Analysis** — scatter plot (listed vs predicted price) + discount distribution
- **Agent Reasoning** — expandable AI analysis cards with risk flags per deal
- **Alert History** — log of every Telegram/email/webhook alert sent

---

## 🏗️ Architecture

```
PakWheels.com
      │
      ▼
 scraper.py ──(optional: MONGODB_URI)──► mongo_store.py (MongoDB raw_listings)
      │
      ▼
 db.py (SQLite — source of truth for cleaned data & dashboard)
      │
      ├───────────────┐
      ▼               ▼
predict.py      deal_detector.py
(statistical    (flags listings ≥ DEAL_THRESHOLD%
 OR QLoRA)       below predicted fair market price)
                      │
                      ▼
                   agent.py
         (Groq API or local Ollama LLM
          scores deals 1-10 & flags risks;
          optional LangSmith tracing)
                      │
        ┌─────────────┴─────────────┐
        ▼ (if N8N_WEBHOOK_URL set)  ▼ (default / fallback)
 webhook_alert.py            telegram_bot.py + email_alert.py
 (n8n workflow)                     │
        └─────────────┬─────────────┘
                      ▼
             dashboard.py (Gradio)
             ├── Tab 1: Live Deals
             ├── Tab 2: Price Analysis
             ├── Tab 3: Agent Reasoning
             └── Tab 4: Alert History

Orchestration: main.py (PIPELINE_ENGINE=legacy default, or PIPELINE_ENGINE=langgraph via graph_pipeline.py)
Automation:    .github/workflows/run_pipeline.yml (every 3 hours, GitHub Actions)
```

---

## 🧩 Optional Agentic & Infrastructure Capabilities

Every capability below is **OPTIONAL and OFF by default** so scheduled GitHub Actions runs behave identically to the baseline pipeline unless explicitly enabled via environment variables. If any optional service is unreachable, the pipeline logs a warning and falls back gracefully.

### 1. LangGraph Orchestration (`PIPELINE_ENGINE=langgraph`)

Set `PIPELINE_ENGINE=langgraph` (default is `legacy`) to run the pipeline as a compiled LangGraph `StateGraph` (`graph_pipeline.py`) with nodes `scrape`, `predict`, `detect_deals`, `evaluate`, and `alert`. Both `scrape` and `evaluate` have a 3-attempt retry policy, and conditional edges stop the graph early when scraping returns 0 listings, when no candidate deals pass `DEAL_THRESHOLD`, or when no deals meet `AGENT_THRESHOLD`:

```mermaid
flowchart TD
    START(["START"]) --> scrape["scrape (RetryPolicy)"]
    scrape -->|"0 listings or error"| END_NODE(["END"])
    scrape -->|"listings > 0"| predict["predict"]
    predict --> detect_deals["detect_deals"]
    detect_deals -->|"0 candidate deals"| END_NODE
    detect_deals -->|"candidate deals > 0"| evaluate["evaluate (RetryPolicy)"]
    evaluate -->|"no deal >= AGENT_THRESHOLD"| END_NODE
    evaluate -->|"qualifying deals > 0"| alert["alert"]
    alert --> END_NODE
```

### 2. LangSmith Tracing (`LANGCHAIN_TRACING_V2=true`)

Enable optional tracing of the LangGraph execution and the AI evaluator's LLM calls using standard LangSmith environment variables only (nothing hardcoded):

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=CarScout_AI
```

- When `LANGCHAIN_TRACING_V2` is unset or `false`, tracing is a complete no-op.
- If `LANGCHAIN_TRACING_V2=true` is set without `LANGCHAIN_API_KEY`, the pipeline logs a warning, disables tracing, and continues normally.
- **Free-Tier Notice**: Check LangSmith's current Developer (free) tier limits at [langchain.com/pricing](https://www.langchain.com/pricing) (typically 5,000 traces/month on the free Developer plan).

### 3. n8n + Webhooks for Alert Delivery (`N8N_WEBHOOK_URL`)

When `N8N_WEBHOOK_URL` is set, `webhook_alert.py` POSTs a JSON payload for each qualifying deal (using the real SQLite column names plus `title`) with a 5-second timeout and retry, and optionally attaches `X-Webhook-Secret` from `N8N_WEBHOOK_SECRET`:
- If the webhook succeeds, direct Telegram/email sending is skipped for that deal so duplicates are never sent.
- If `N8N_WEBHOOK_URL` is unset or the webhook request fails, the pipeline automatically falls back to `telegram_bot.py` and `email_alert.py`.
- See [`n8n/README.md`](n8n/README.md) and [`n8n/carscout-alerts.json`](n8n/carscout-alerts.json) for import instructions and a sample `curl` test command. Note that GitHub Actions cannot reach a local `localhost:5678` n8n instance, so webhook delivery is designed for local runs.

### 4. MongoDB Raw Listing Store (`MONGODB_URI`)

When `MONGODB_URI` is set, `mongo_store.py` saves each scraped listing's raw JSON (including `scraped_at` and `source_url`) to MongoDB (`MONGODB_DB=carscout`, `MONGODB_COLLECTION=raw_listings`) using `pymongo` with an upsert on `listing_id` so repeated runs never create duplicates. SQLite (`gaadiscout.db`) remains the source of truth for cleaned data and the dashboard.

### 5. Ollama Local LLM Provider (`AGENT_PROVIDER=ollama`)

Set `AGENT_PROVIDER=ollama` along with `OLLAMA_BASE_URL` (e.g., `http://localhost:11434`) and `OLLAMA_MODEL` (no model name is hardcoded) to run deal evaluation in `agent.py` against a local open-source model via Ollama instead of Groq. It uses the exact same system prompt and `DealEvaluation` JSON output schema.

### 6. Optional Local Docker Compose Profiles (`docker-compose.yml`)

A `docker-compose.yml` is provided for local development with `n8n`, `mongo`, and `ollama` each isolated in its own Compose profile so nothing starts by default:

```bash
# Start only n8n (http://localhost:5678)
docker compose --profile n8n up -d

# Start only MongoDB (mongodb://localhost:27017)
docker compose --profile mongo up -d

# Start only Ollama (http://localhost:11434)
docker compose --profile ollama up -d

# Start all three local services together
docker compose --profile n8n --profile mongo --profile ollama up -d
```

---

## 🛠️ Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| Scraping | `requests` + `BeautifulSoup4` (`lxml`) | Free |
| Orchestration | Python (`legacy`) or `LangGraph` (`langgraph`) | Free (open-source) |
| Tracing (optional) | `LangSmith` | Free tier (check monthly trace limits) |
| Database (primary) | SQLite (`gaadiscout.db`) | Free (built-in) |
| Raw Store (optional) | MongoDB + `pymongo` | Free (local Docker / self-hosted) |
| Price prediction | Statistical median + optional QLoRA fine-tuning | Free |
| Fine-tuning | HuggingFace `transformers`, `peft`, `trl`, Colab GPU | Free |
| AI Evaluator | Groq API (`llama-3.1-8b-instant`) or local `Ollama` | Free tier / Free local |
| Alerts | n8n Webhook (optional) + Telegram Bot API + Gmail SMTP | Free |
| Dashboard | Gradio (`dashboard.py` / `app.py`) | Free |
| Hosting | Hugging Face Spaces | Free |
| Automation | GitHub Actions | Free (2,000 min/month on free accounts) |

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/MahadAsif1133/CarScout_AI.git
cd CarScout_AI
pip install -r requirements.txt
# Optional dev/test tools (pytest, ruff):
pip install -r requirements-dev.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys (see "Getting Free API Keys" section below)
```

### 3. Run the pipeline once

```bash
python main.py
# Or run with the LangGraph engine:
PIPELINE_ENGINE=langgraph python main.py
```

### 4. Launch the dashboard

```bash
python dashboard.py
# Open http://localhost:7860
```

### 5. (Optional) Run individual modules for testing

```bash
python db.py            # Initialise DB + insert test row
python scraper.py       # Scrape 2 pages, print sample listing
python predict.py       # Test price prediction
python deal_detector.py # Detect deals in DB
python agent.py         # Evaluate a mock deal with AI
python telegram_bot.py  # Send a test Telegram message
python email_alert.py   # Send a test email
```

---

## 🔑 Getting Free API Keys

### Groq API (AI Agent)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free (no credit card required)
3. Go to **API Keys** → **Create API Key**
4. Add to `.env`: `GROQ_API_KEY=your_key`

**Free tier:** 14,400 requests/day with `llama-3.1-8b-instant` ✅

---

### Telegram Bot
1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy the bot token to `.env`: `TELEGRAM_BOT_TOKEN=...`
4. Start a chat with your new bot
5. Get your Chat ID by messaging `@userinfobot` on Telegram
6. Add to `.env`: `TELEGRAM_CHAT_ID=your_chat_id`

**Free tier:** Unlimited messages ✅

---

### Gmail SMTP (Email Alerts)
1. Enable 2-Factor Authentication on your Google account
2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Create an App Password for "Mail" → "Windows Computer" (or any)
4. Copy the 16-character password (no spaces)
5. Add to `.env`:
   ```
   EMAIL_SENDER=your@gmail.com
   EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
   EMAIL_RECIPIENT=recipient@example.com
   ```

**Free tier:** 500 emails/day

---

### Hugging Face Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → Type: **Read**
3. Add to `.env`: `HF_TOKEN=hf_...`

Needed for: downloading models, pushing your fine-tuned adapter.
**Free tier:** Unlimited downloads of public models

---

## 🤖 QLoRA Fine-Tuning (Optional)

The price predictor works **out-of-the-box** using statistical median prediction (`PREDICTION_MODE=statistical` in GitHub Actions, or `PREDICTION_MODE=auto` locally which uses `./lora_adapter/` only if present).
Once you have scraped enough data (100+ listings), you can fine-tune a small LLM for better predictions using **Google Colab's free GPU**.

### Steps:
1. Run the pipeline a few times to collect listings: `python main.py`
2. Upload `gaadiscout.db` to Google Colab: `/content/gaadiscout.db`
3. Open `train_model.py` in Colab and run the cells in order
4. After training, download `./lora_adapter/` to your project root
5. Set `PREDICTION_MODE=finetuned` in `.env`

**Recommended base models (both free on HF Hub):**
- `Qwen/Qwen2.5-1.5B-Instruct` (1.5B params, ~2GB VRAM)
- `meta-llama/Llama-3.2-1B-Instruct` (1B params, requires HF token)

---

## 🚀 Deploying to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Visibility: Public or Private
2. Push your code:
   ```bash
   git remote add hf https://huggingface.co/spaces/your-username/carscout-ai
   git push hf main
   ```
3. Add Secrets in Space Settings → Repository Secrets:
   - `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - `EMAIL_SENDER`, `EMAIL_APP_PASSWORD`, `EMAIL_RECIPIENT`
   - `HF_TOKEN`

The `app.py` file is the Spaces entry point.

---

## ⏰ Automating with GitHub Actions

The workflow `.github/workflows/run_pipeline.yml` (named **GaadiScout Pipeline**) runs automatically every 3 hours.

### Setup:
1. Push code to GitHub
2. Go to **Settings → Secrets and variables → Actions**
3. Add these secrets:
   - `GROQ_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `EMAIL_SENDER`
   - `EMAIL_APP_PASSWORD`
   - `EMAIL_RECIPIENT`
   - `HF_TOKEN`
4. Enable Actions in the **Actions** tab

### Manual trigger (Dry Run):
Go to **Actions** → **GaadiScout Pipeline** → **Run workflow** →
Set `disable_alerts` to `true` (and optionally adjust `max_pages` or `deal_threshold`) to test the workflow with alerts disabled.

> **Note:** GitHub Actions free tier gives 2,000 minutes/month for private repos (and is free for public repos).
> Running every 3 hours = 8 runs/day × ~3 min/run = 24 min/day = ~720 min/month. ✅

---

## ⚙️ Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SCRAPE_MAX_PAGES` | `5` | Max pages to scrape per run (~24 listings/page) |
| `SCRAPE_DETAILS` | `false` | Also fetch each listing's detail page (`true`/`false`, reserved in scraper) |
| `DEAL_THRESHOLD` | `15.0` | Min % below predicted price to flag as deal |
| `AGENT_THRESHOLD` | `7` | Min agent score (1-10) to send alerts |
| `PREDICTION_MODE` | `auto` | `auto` / `statistical` / `finetuned` (`statistical` in GitHub Actions) |
| `ENABLE_TELEGRAM` | `true` | Enable/disable Telegram alerts |
| `ENABLE_EMAIL` | `true` | Enable/disable email alerts |
| `GROQ_API_KEY` | *(empty)* | Groq API key for LLM deal evaluation |
| `AGENT_MODEL` | `llama-3.1-8b-instant` | Groq model for deal evaluation |
| `AGENT_PROVIDER` | `groq` | LLM provider for `agent.py`: `groq` (default) or `ollama` |
| `OLLAMA_BASE_URL` | *(empty)* | Base URL for local Ollama server (e.g. `http://localhost:11434`) |
| `OLLAMA_MODEL` | *(empty)* | Model name to request from Ollama (required when `AGENT_PROVIDER=ollama`) |
| `PIPELINE_ENGINE` | `legacy` | Orchestration engine: `legacy` (default) or `langgraph` |
| `LANGCHAIN_TRACING_V2` | `false` | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | *(empty)* | LangSmith API key (used when `LANGCHAIN_TRACING_V2=true`) |
| `LANGCHAIN_PROJECT` | `CarScout_AI` | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith API endpoint |
| `N8N_WEBHOOK_URL` | *(empty)* | Optional n8n webhook URL for alert delivery (falls back to Telegram/Email) |
| `N8N_WEBHOOK_SECRET` | *(empty)* | Optional shared secret sent in `X-Webhook-Secret` header |
| `MONGODB_URI` | *(empty)* | Optional MongoDB connection string for archiving raw scraped listings |
| `MONGODB_DB` | `carscout` | MongoDB database name when `MONGODB_URI` is set |
| `MONGODB_COLLECTION` | `raw_listings` | MongoDB collection name when `MONGODB_URI` is set |
| `BASE_MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model for QLoRA fine-tuning |
| `HF_TOKEN` | *(empty)* | Hugging Face token for gated models / Hub uploads |
| `PORT` | `7860` | Port for running `dashboard.py` directly |

---

## 🛡️ Ethical Scraping Policy

CarScout_AI respects PakWheels.com's `robots.txt`:
- ✅ Only scrapes **public search result pages** (`/used-cars/search/-/`)
- ✅ Uses a descriptive **User-Agent** identifying the bot and its purpose
- ✅ Adds **1–2.5 second random delays** between all requests
- ✅ Never requests disallowed paths (`/notify_seller`, `/manage/`, `?sortby=`, etc.)
- ✅ Handles pagination politely, stops at the last page
- ⚠️ For production use, consider caching aggressively and limiting run frequency

---

## 📁 Project Structure

```
CarScout_AI/
├── scraper.py              # PakWheels JSON-LD / HTML scraper
├── db.py                   # SQLite database layer (gaadiscout.db)
├── mongo_store.py          # Optional MongoDB raw listing store
├── predict.py              # Hierarchical statistical + QLoRA fair-price predictor
├── train_model.py          # QLoRA fine-tuning script (run on Colab)
├── deal_detector.py        # Candidate deal flagging logic
├── agent.py                # Single LLM deal evaluator (Groq or Ollama + heuristic flags)
├── webhook_alert.py        # Optional n8n webhook alert sender with fallback
├── telegram_bot.py         # Telegram Bot API alert sender
├── email_alert.py          # Gmail SMTP HTML alert sender
├── graph_pipeline.py       # LangGraph StateGraph orchestrator
├── main.py                 # Pipeline entry point (legacy default or langgraph)
├── dashboard.py            # Gradio 4-tab live dashboard
├── app.py                  # Hugging Face Spaces entry point
├── docker-compose.yml      # Optional local profiles for n8n, mongo, and ollama
├── n8n/
│   ├── carscout-alerts.json # Importable n8n workflow (no embedded credentials)
│   └── README.md           # n8n setup and curl test instructions
├── tests/                  # Pytest unit test suite (all mocked)
├── requirements.txt        # Production Python dependencies
├── requirements-dev.txt    # Development dependencies (pytest, ruff)
├── .env.example            # Environment variable template
└── .github/
    └── workflows/
        └── run_pipeline.yml # GitHub Actions cron + manual dry-run workflow
```

---

## 📄 License

MIT License — free for personal and educational use.

---

*Built with ❤️ using open-source tools. Not affiliated with PakWheels.com.*
