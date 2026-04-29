# BR-PPO Alpaca Paper Trading System

Fully automated daily paper trading using a **Proximal Policy Optimization (PPO)** agent to dynamically allocate across equity portfolios and ETFs via the Alpaca Paper Trading API.

---

## Architecture

```
GitHub Actions (cron: Mon–Fri 9 AM ET)
    │
    ├── 1. Download PPO model from Google Drive
    ├── 2. Run paper_trader.py  →  strategy_engine.py
    │        ├── Download prices (yfinance)
    │        ├── Build features (returns, Sharpe, drawdown, Ichimoku)
    │        ├── PPO inference → action → target weights
    │        ├── Connect to Alpaca → get account/positions
    │        ├── Build order plan (rebalance deltas)
    │        └── Submit market orders at open
    ├── 3. Commit logs/ CSVs back to this repo
    └── 4. Streamlit Cloud reads logs → live dashboard
```

---

## One-Time Setup

### Step 1 — Create GitHub Repo

1. Go to [github.com/new](https://github.com/new)
2. Name it `br-ppo-paper-trading` (or anything you like)
3. Set to **Private** (recommended)
4. **Do not** initialize with README (you'll push these files)
5. Push these files:
   ```bash
   cd br_ppo_trading/
   git init
   git add .
   git commit -m "Initial commit — BR-PPO paper trading system"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/br-ppo-paper-trading.git
   git push -u origin main
   ```

---

### Step 2 — Share Google Drive Model Files

The GitHub Actions runner downloads your PPO model from Google Drive using `gdown`.
**You must make both files publicly accessible** (anyone with the link):

1. Open Google Drive → find `v10_bro_ppo_allocation_agent.zip`
2. Right-click → **Share** → **Change to Anyone with the link** → Viewer
3. Do the same for `v10_bro_ppo_allocation_agent_metadata.json`

The file IDs are already pre-filled in `.env.template` and the workflow.

---

### Step 3 — Create a GitHub Personal Access Token (PAT)

The workflow commits log files back to the repo after each run. It needs a PAT with `repo` scope.

1. Go to [github.com/settings/tokens/new](https://github.com/settings/tokens/new)
2. Note: **`BR-PPO Trading Bot`**
3. Expiration: **No expiration** (or 1 year)
4. Scopes: ✅ `repo` (full control)
5. Click **Generate token** — copy the token value

---

### Step 4 — Add GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:

| Secret Name              | Value                                    |
|--------------------------|------------------------------------------|
| `GH_PAT`                 | Your Personal Access Token from Step 3  |
| `ALPACA_API_KEY`         | Your Alpaca Paper API Key               |
| `ALPACA_SECRET_KEY`      | Your Alpaca Paper Secret Key            |
| `GDRIVE_MODEL_FILE_ID`   | `1qPvDVLU68AiSWDDo9lvoB1rfGZ7WAkQM`    |
| `GDRIVE_METADATA_FILE_ID`| `1j5HWvyTY7O7XW2mc0NhrKtP09QiM1nnt`    |

> **Where to get Alpaca paper keys:**
> Log into [alpaca.markets](https://alpaca.markets) → Paper Trading → API Keys → Generate

---

### Step 5 — Deploy the Dashboard to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Repository: `YOUR_USERNAME/br-ppo-paper-trading`
5. Branch: `main`
6. Main file path: `streamlit_app.py`
7. Click **Deploy**

The dashboard will be live at `https://YOUR_APP.streamlit.app` and automatically shows the latest logs committed by GitHub Actions.

---

### Step 6 — Test with a Manual Run

1. Go to your repo → **Actions** tab
2. Click **Daily Paper Trader**
3. Click **Run workflow** → set `dry_run: true` for a safe first test
4. Watch the logs — you should see the PPO action, target weights, and planned orders
5. Check the dashboard — it should now show data!

Once you've confirmed everything looks right, flip `BRPPO_SUBMIT_ORDERS=True` (already the default in the workflow for production runs).

---

## Schedule

The workflow runs automatically:
- **When:** Mon–Fri at **13:00 UTC** (= 9:00 AM ET summer / 8:00 AM ET winter)
- **Before market open** (NYSE opens at 9:30 AM ET), so market orders fill at the open bell
- **Market holidays** are automatically detected and skipped

You can also trigger it manually anytime from the **Actions** tab.

---

## File Structure

```
br-ppo-paper-trading/
├── strategy_engine.py          # Core: data, features, PPO inference, orders
├── paper_trader.py             # Entry point with fallback handling
├── streamlit_app.py            # Dashboard (Streamlit Cloud)
├── requirements.txt            # All dependencies
├── .env.template               # Local dev config template
├── .gitignore
│
├── scripts/
│   ├── download_model.py       # Downloads model from Google Drive via gdown
│   └── market_check.py         # NYSE trading day checker
│
├── .github/workflows/
│   └── daily_paper_trader.yml  # GitHub Actions schedule
│
├── artifacts/
│   └── .gitkeep               # Model files downloaded here at runtime
│
└── logs/                       # Committed by GitHub Actions after each run
    ├── decisions/
    │   ├── decisions.csv        # Full history
    │   └── latest_decision.csv
    ├── portfolio/
    │   └── portfolio.csv
    ├── target_weights/
    │   ├── target_weights.csv
    │   └── latest_target_weights.csv
    ├── positions/
    │   └── latest_positions.csv
    └── orders/
        ├── planned_orders.csv
        ├── submitted_orders.csv
        ├── latest_planned_orders.csv
        └── latest_submitted_orders.csv
```

---

## Trading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BRPPO_SUBMIT_ORDERS` | `True` | Set `False` for dry run |
| `BRPPO_MIN_TRADE_DOLLARS` | `$25` | Min order size |
| `BRPPO_MAX_POSITION_WEIGHT` | `20%` | Max single position |
| `BRPPO_MAX_GROSS_EXPOSURE` | `100%` | Max portfolio exposure |
| `BRPPO_CASH_BUFFER_PCT` | `2%` | Cash reserve |
| `BRPPO_REBALANCE_THRESHOLD_WEIGHT` | `0.5%` | Min weight drift to trigger trade |
| `BRPPO_V6_MAX_NAMES` | `30` | Max stocks in V6 Alpha basket |

---

## Actions the PPO Agent Can Choose

| Action | Description |
|--------|-------------|
| `current_ew` | 100% equal-weight S&P 500 proxy (RSP) |
| `top_ew` | 100% top-decile equal-weight (RSP) |
| `v6_alpha` | 100% V6 Alpha stock basket (top 30 by momentum/Ichimoku) |
| `v8_blend` | V8 blend: 20% V6 + 50% EW + 30% QQQ |
| `current70_v6_30` | 70% EW + 30% V6 Alpha |
| `bil_cash` | 100% cash (BIL) |
| `spy70_bil30` | 70% SPY + 30% BIL (defensive) |
| … | 15 total allocation strategies |

---

## Local Development

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/br-ppo-paper-trading.git
cd br-ppo-paper-trading

# Install deps
pip install -r requirements.txt

# Set up .env
cp .env.template .env
# Edit .env with your Alpaca keys

# Download model
python scripts/download_model.py

# Dry run (no orders submitted)
BRPPO_SUBMIT_ORDERS=False python paper_trader.py

# Launch dashboard locally
streamlit run streamlit_app.py
```

---

## Safety Features

- **Dry-run protection:** `BRPPO_SUBMIT_ORDERS=False` prevents any orders
- **Fallback safety:** If the live basket fails, falls back to ETF proxies without submitting unless `BRPPO_ALLOW_FALLBACK_ORDERS=True`
- **Holiday skip:** Automatically detects NYSE holidays and skips
- **Manual override:** Trigger with `dry_run: true` from GitHub Actions UI anytime
- **Log backup:** Every run's logs are also uploaded as GitHub Actions artifacts (90-day retention)
