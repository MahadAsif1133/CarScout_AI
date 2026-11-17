# n8n Webhook Alert Workflow (`carscout-alerts.json`)

This directory contains an importable, open-source **n8n** workflow (`carscout-alerts.json`) that receives deal alerts from `CarScout_AI` (`webhook_alert.py`) via HTTP POST and dispatches formatted notifications to both **Telegram** and **Gmail/SMTP** without embedding any credentials in the file.

> **Important — Local Runs vs. GitHub Actions:**
> GitHub Actions runners execute on GitHub-hosted cloud VMs and **cannot reach a local `localhost:5678` n8n instance**. Therefore, `N8N_WEBHOOK_URL` is intended for **local runs** (or if you expose a self-hosted tunnel). When `N8N_WEBHOOK_URL` is unset (as in GitHub Actions) or if the webhook is unreachable, `CarScout_AI` logs a warning and automatically falls back to direct `telegram_bot.py` and `email_alert.py` delivery so no alert is ever lost.

---

## Workflow Structure

1. **Webhook Trigger** (`POST /webhook/carscout-alerts`) — receives the deal JSON payload.
2. **Check Webhook Secret** (`IF` node) — compares the `X-Webhook-Secret` request header against `$env.N8N_WEBHOOK_SECRET` (if configured).
3. **Telegram Deal Alert** — formats and sends the deal message to `$env.TELEGRAM_CHAT_ID`.
4. **Send SMTP Email** — sends an HTML email alert to `$env.EMAIL_RECIPIENT`.

---

## Import & Setup Steps

1. **Start n8n locally** (using the optional Docker Compose profile in the repo root):
   ```bash
   docker compose --profile n8n up -d
   ```
   Open `http://localhost:5678` in your browser.

2. **Import the workflow**:
   - In n8n, click **Workflows** → **Add Workflow** → **Import from File…**
   - Select `n8n/carscout-alerts.json`.

3. **Create Credentials inside n8n**:
   - Open the **Telegram Deal Alert** node → create a new **Telegram API** credential with your `TELEGRAM_BOT_TOKEN`.
   - Open the **Send SMTP Email** node → create a new **SMTP** credential (`smtp.gmail.com`, port `465`, SSL enabled, your Gmail address, and 16-character App Password).

4. **Activate the workflow**:
   - Toggle the workflow to **Active** in the top-right corner of n8n.
   - Set these variables in your local `.env`:
     ```env
     N8N_WEBHOOK_URL=http://localhost:5678/webhook/carscout-alerts
     N8N_WEBHOOK_SECRET=your_shared_secret
     ```

---

## Testing with `curl`

Run the following `curl` command to send a sample deal payload (using the exact database column names) to your local n8n webhook:

```bash
curl -X POST http://localhost:5678/webhook-test/carscout-alerts \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: your_shared_secret" \
  -d '{
    "title": "Toyota Corolla 2019",
    "listing_url": "https://www.pakwheels.com/used-cars/toyota-corolla-2019-for-sale-in-lahore-123456",
    "make": "Toyota",
    "model": "Corolla",
    "year": 2019,
    "mileage": 55000,
    "city": "Lahore",
    "price": 3100000,
    "registration_city": "Lahore",
    "transmission": "Automatic",
    "engine_type": "Petrol 1600cc",
    "description": "First owner, total genuine paint, complete documents available.",
    "posted_date": "Updated 10 minutes ago",
    "image_url": "https://cache1.pakwheels.com/ad_pictures/sample.jpg",
    "scraped_at": "2026-11-17T12:00:00",
    "predicted_price": 3800000,
    "price_gap_pct": 18.42,
    "deal_score": 8,
    "agent_summary": "Well-maintained first-owner Corolla priced 18.4% below fair value with no risk flags.",
    "risk_flags": [],
    "prediction_method": "exact_model_year",
    "prediction_level": 2,
    "confidence_text": "Based on 12 similar listings (high confidence)"
  }'
```

*(Use `/webhook/carscout-alerts` when the workflow is Active, or `/webhook-test/carscout-alerts` when clicking "Listen for test event" in the n8n editor.)*
