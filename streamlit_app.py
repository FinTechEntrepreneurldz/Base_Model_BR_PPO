"""
BR-PPO Alpaca Paper Trading Dashboard
Streamlit Cloud deployment — reads committed log CSVs from the repo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BR-PPO Paper Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_ROOT  = Path(__file__).parent
LOGS_ROOT  = REPO_ROOT / "logs"
MODELS_YAML = REPO_ROOT / "models.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_models_registry():
    """Read models.yaml; return list of enabled models with id, name, color, etc.

    Falls back gracefully if pyyaml isn't available or the file is missing.
    """
    try:
        import yaml
        with open(MODELS_YAML) as f:
            data = yaml.safe_load(f) or {}
        enabled = [m for m in (data.get("models") or []) if m.get("enabled", True)]
        if enabled:
            return enabled
    except Exception:
        pass

    # Fallback: derive from logs/ subdirectories
    fallback = []
    if LOGS_ROOT.exists():
        for sub in sorted(LOGS_ROOT.iterdir()):
            if sub.is_dir() and (sub / "decisions").exists():
                fallback.append({"id": sub.name, "name": sub.name, "color": "#4c9eff", "environment": sub.name})
    if fallback:
        return fallback

    # Last-resort legacy fallback
    return [{"id": "default", "name": "Default", "color": "#4c9eff", "environment": "default"}]

PALETTE = {
    "bg":       "#0e1117",
    "card":     "#1a1f2e",
    "green":    "#00d4aa",
    "red":      "#ff4b6e",
    "blue":     "#4c9eff",
    "gold":     "#ffd166",
    "text":     "#e8eaf0",
    "muted":    "#8892a4",
}

ACTION_COLORS = {
    "current_ew":             "#4c9eff",
    "top_ew":                 "#00d4aa",
    "spy":                    "#ffd166",
    "v6_alpha":               "#a78bfa",
    "v8_blend":               "#f97316",
    "bil_cash":               "#8892a4",
}


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"]          { background: #12161f; border-right: 1px solid #1e2535; }
.stTabs [data-baseweb="tab-list"]  { background: #12161f; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"]       { color: #8892a4; border-radius: 6px; }
.stTabs [aria-selected="true"]     { background: #1a1f2e !important; color: #e8eaf0 !important; }

/* Metric cards */
.metric-card {
    background: #1a1f2e;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-label  { font-size: 12px; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value  { font-size: 28px; font-weight: 700; color: #e8eaf0; }
.metric-delta  { font-size: 13px; margin-top: 4px; }
.delta-pos     { color: #00d4aa; }
.delta-neg     { color: #ff4b6e; }
.delta-neu     { color: #8892a4; }

/* Action badge */
.action-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    background: #1e2535;
    color: #4c9eff;
    border: 1px solid #4c9eff40;
}

/* Status dot */
.status-live  { color: #00d4aa; font-weight: 600; }
.status-dry   { color: #ffd166; font-weight: 600; }
.status-error { color: #ff4b6e; font-weight: 600; }

/* Section header */
.section-header {
    font-size: 16px;
    font-weight: 600;
    color: #e8eaf0;
    border-bottom: 1px solid #1e2535;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Divider */
hr { border-color: #1e2535; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)   # Re-read from disk every 2 minutes
def load_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_all(log_dir: Path):
    log_dir = Path(log_dir)
    return {
        "decisions":        load_csv(log_dir / "decisions"      / "decisions.csv"),
        "latest_decision":  load_csv(log_dir / "decisions"      / "latest_decision.csv"),
        "portfolio":        load_csv(log_dir / "portfolio"       / "portfolio.csv"),
        "target_weights":   load_csv(log_dir / "target_weights"  / "latest_target_weights.csv"),
        "tw_history":       load_csv(log_dir / "target_weights"  / "target_weights.csv"),
        "positions":        load_csv(log_dir / "positions"       / "latest_positions.csv"),
        "planned_orders":   load_csv(log_dir / "orders"          / "latest_planned_orders.csv"),
        "submitted_orders": load_csv(log_dir / "orders"          / "latest_submitted_orders.csv"),
        "orders_history":   load_csv(log_dir / "orders"          / "submitted_orders.csv"),
        "signal_history":   load_csv(log_dir / "health"          / "signal_history.csv"),
        "health_status":    _load_health_status(log_dir),
    }


def _load_health_status(log_dir: Path):
    path = Path(log_dir) / "health" / "health_status.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Multi-repo loaders — fetch a model's logs from raw.githubusercontent.com
# ──────────────────────────────────────────────────────────────────────────────
# Each model in models.yaml may live in its own GitHub repo. The dashboard reads
# its committed log CSVs/JSON directly from raw.githubusercontent.com (no auth
# needed for public repos). Falls back to the local repo for legacy models that
# don't specify a `repo` field in the registry.

RAW_BASE = "https://raw.githubusercontent.com"


def _model_logs_url(model: dict, relpath: str) -> str | None:
    """Build the raw.githubusercontent.com URL for a model's log file.

    Returns None if the model has no `repo` field (caller should use local path).
    """
    repo = model.get("repo")
    if not repo:
        return None
    branch = model.get("branch", "main")
    logs_path = (model.get("logs_path") or f"logs/{model['id']}").strip("/")
    relpath = relpath.strip("/")
    return f"{RAW_BASE}/{repo}/{branch}/{logs_path}/{relpath}"


@st.cache_data(ttl=120, show_spinner=False)
def _load_csv_url(url: str) -> pd.DataFrame:
    """Fetch a CSV from a URL via requests (more reliable than pd.read_csv on
    Streamlit Cloud), parse with pandas, return empty DataFrame on any failure."""
    try:
        import io
        import requests
        # Cache-busting User-Agent and explicit no-cache so Streamlit Cloud
        # doesn't get a stale CDN response after a fresh push to the model repo.
        headers = {
            "User-Agent": "streamlit-dashboard/1.0",
            "Cache-Control": "no-cache",
        }
        r = requests.get(url, timeout=15, headers=headers)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def _load_json_url(url: str) -> dict:
    """Fetch a JSON document from a URL. Returns {} on any failure."""
    try:
        import requests
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def load_model_csv(model: dict, relpath: str) -> pd.DataFrame:
    """Load a CSV under a model's logs_path (URL if `repo` set, else local file)."""
    url = _model_logs_url(model, relpath)
    if url is not None:
        return _load_csv_url(url)
    # Local fallback (legacy: model lives in this same repo, no repo field)
    logs_path = model.get("logs_path") or f"logs/{model['id']}"
    return load_csv(REPO_ROOT / logs_path / relpath)


def load_model_health(model: dict) -> dict:
    """Load a model's health_status.json from the CENTRAL location.

    Per the Option 2 architecture (see scripts/central_health_monitor.py),
    health metrics for ALL models — even those whose trading logs live in
    separate GitHub repos — are computed and stored centrally in this
    dashboard repo at logs/<model_id>/health/health_status.json.

    This single source of truth means the dashboard never needs to fetch
    health from a model's own repo, eliminating per-repo schema drift.
    """
    central_path = REPO_ROOT / "logs" / model["id"] / "health" / "health_status.json"
    if central_path.exists():
        return _load_health_status(central_path.parent.parent)
    return {}


def _interpret_health(health_dict):
    """Normalize a model's health_status.json into a (icon, color, label, is_pending) tuple.

    The dashboard expects only {healthy, warning, degraded} as real statuses; any other
    value (unknown, hold, empty, missing file) is treated as 'Pending first health check'.

    'Pending' specifically means: the model exists but hasn't accumulated enough trading
    days for a meaningful Sharpe / IC / drift signal (default threshold: n_decisions >= 5).
    """
    if not health_dict:
        return ("\U0001f550", "#8892a4", "Pending first check", True)
    overall = (health_dict or {}).get("overall_status", "unknown")
    n_decisions = (health_dict or {}).get("n_decisions") or 0
    # Insufficient history → always Pending regardless of stored status
    if n_decisions < 5:
        return ("\U0001f550", "#8892a4", "Pending first check", True)
    # Recognized real statuses
    real = {
        "healthy":  ("\U0001f7e2", "#00d4aa", "Healthy",  False),
        "warning":  ("\U0001f7e1", "#ffd166", "Warning",  False),
        "degraded": ("\U0001f534", "#ff4b6e", "Degraded", False),
    }
    if overall in real:
        return real[overall]
    return ("\U0001f550", "#8892a4", "Pending first check", True)


def _normalize_portfolio_columns(df):
    """Guarantee a `portfolio_value` column regardless of source schema.

    Different paper traders use different column names for the equity series:
      - Model A (Base_Model_BR_PPO): writes 'portfolio_value'
      - Model B / Model C (separate repos): writes 'equity'
    The dashboard expects 'portfolio_value' everywhere, so we alias here.
    """
    import pandas as _pd
    if df is None or not hasattr(df, "columns") or df.empty:
        return df
    if "portfolio_value" in df.columns:
        return df
    for alt in ("equity", "account_value", "total_equity"):
        if alt in df.columns:
            df = df.copy()
            df["portfolio_value"] = _pd.to_numeric(df[alt], errors="coerce")
            return df
    return df


def load_all_for_model(model: dict) -> dict:
    """Mirror of load_all() but for a model dict (works across repos)."""
    return {
        "decisions":        load_model_csv(model, "decisions/decisions.csv"),
        "latest_decision":  load_model_csv(model, "decisions/latest_decision.csv"),
        "portfolio":        _normalize_portfolio_columns(load_model_csv(model, "portfolio/portfolio.csv")),
        "target_weights":   load_model_csv(model, "target_weights/latest_target_weights.csv"),
        "tw_history":       load_model_csv(model, "target_weights/target_weights.csv"),
        "positions":        load_model_csv(model, "positions/latest_positions.csv"),
        "planned_orders":   load_model_csv(model, "orders/latest_planned_orders.csv"),
        "submitted_orders": load_model_csv(model, "orders/latest_submitted_orders.csv"),
        "orders_history":   load_model_csv(model, "orders/submitted_orders.csv"),
        "signal_history":   load_model_csv(model, "health/signal_history.csv"),
        "health_status":    load_model_health(model),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def fmt_dollars(v):
    try: return f"${float(v):,.2f}"
    except: return "—"

def fmt_pct(v):
    try: return f"{float(v)*100:.2f}%"
    except: return "—"

def hex_to_rgba(hex_color, alpha=0.08):
    """Convert #rrggbb hex to rgba() string for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ──────────────────────────────────────────────────────────────────────────────
# Performance statistics — mirrors notebook perf_stats() exactly
# ──────────────────────────────────────────────────────────────────────────────

def compute_perf_stats(daily_returns: pd.Series, min_obs: int = 5) -> dict:
    """
    Comprehensive performance statistics for a series of daily returns.

    Returns a dict with: total_return, ann_return, ann_vol, sharpe, sortino,
    calmar, max_dd, hit_rate, t_stat, profit_factor, avg_win, avg_loss,
    win_loss_ratio, best_day, worst_day, n_days, n_win_days, n_loss_days,
    longest_dd_days, current_dd, current_dd_days.
    """
    nan_result = dict(
        total_return=np.nan, ann_return=np.nan, ann_vol=np.nan,
        sharpe=np.nan, sortino=np.nan, calmar=np.nan, max_dd=np.nan,
        hit_rate=np.nan, t_stat=np.nan, profit_factor=np.nan,
        avg_win=np.nan, avg_loss=np.nan, win_loss_ratio=np.nan,
        best_day=np.nan, worst_day=np.nan,
        n_days=0, n_win_days=0, n_loss_days=0,
        longest_dd_days=0, current_dd=np.nan, current_dd_days=0,
    )

    x = pd.Series(daily_returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < min_obs:
        nan_result["n_days"] = len(x)
        return nan_result

    eq = (1.0 + x).cumprod()
    total_return = float(eq.iloc[-1] - 1.0)
    ann_return = float(eq.iloc[-1] ** (252.0 / len(x)) - 1.0)
    ann_vol = float(x.std() * np.sqrt(252)) if x.std() > 0 else np.nan
    sharpe = float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan

    downside = x[x < 0].std() * np.sqrt(252) if (x < 0).any() else np.nan
    sortino = float(x.mean() * 252 / downside) if downside and downside > 0 else np.nan

    t_stat = float(x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else np.nan

    dd_series = (eq / eq.cummax() - 1.0)
    max_dd = float(dd_series.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else np.nan

    hit_rate = float((x > 0).mean())
    n_win_days = int((x > 0).sum())
    n_loss_days = int((x < 0).sum())

    wins = x[x > 0]
    losses = x[x < 0]
    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(losses.mean()) if len(losses) else np.nan
    win_loss_ratio = float(abs(avg_win / avg_loss)) if (avg_loss and not pd.isna(avg_loss) and avg_loss != 0) else np.nan
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan

    best_day = float(x.max())
    worst_day = float(x.min())

    # Drawdown duration: longest run where dd_series < 0
    in_dd = dd_series < 0
    longest_dd = 0
    cur_run = 0
    for v in in_dd.values:
        if v:
            cur_run += 1
            longest_dd = max(longest_dd, cur_run)
        else:
            cur_run = 0

    # Current drawdown (from latest peak)
    current_dd = float(dd_series.iloc[-1])
    # Days since last all-time high
    rev = dd_series.iloc[::-1]
    current_dd_days = 0
    for v in rev.values:
        if v < 0:
            current_dd_days += 1
        else:
            break

    return dict(
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_dd=max_dd,
        hit_rate=hit_rate,
        t_stat=t_stat,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        best_day=best_day,
        worst_day=worst_day,
        n_days=len(x),
        n_win_days=n_win_days,
        n_loss_days=n_loss_days,
        longest_dd_days=longest_dd,
        current_dd=current_dd,
        current_dd_days=current_dd_days,
    )


@st.cache_data(ttl=900)  # 15 min cache for benchmark prices
def fetch_benchmark_returns(ticker: str, start: str, end: str = None) -> pd.Series:
    """
    Download benchmark daily returns from yfinance, indexed by NAIVE calendar date
    (no timezone, no time-of-day) so it can be safely aligned with portfolio runs
    that happen at any time during the trading day.
    """
    try:
        import yfinance as yf
        kwargs = dict(start=start, auto_adjust=True, progress=False)
        if end:
            kwargs["end"] = end
        data = yf.download(ticker, **kwargs)
        if data is None or data.empty:
            return pd.Series(dtype=float)

        # Handle BOTH flat columns (old yfinance) and MultiIndex columns (new yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                close = data[("Close", ticker)]
            except KeyError:
                # fallback: any column with "Close" in its top level
                lvl0 = data.columns.get_level_values(0)
                if "Close" in lvl0:
                    close = data["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                else:
                    close = data.iloc[:, 0]
        else:
            close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.Series(close).dropna().astype(float)
        if close.empty:
            return pd.Series(dtype=float)

        # Normalise to naive calendar date (drop tz + time component)
        idx = pd.to_datetime(close.index)
        try:
            idx = idx.tz_convert(None)  # tz-aware → naive
        except (TypeError, AttributeError):
            try:
                idx = idx.tz_localize(None)  # naive but with tz info? strip
            except (TypeError, AttributeError):
                pass
        close.index = pd.to_datetime(idx).normalize()

        rets = close.pct_change().dropna()
        rets.name = ticker
        return rets
    except Exception:
        return pd.Series(dtype=float)


def _to_date_index(s: pd.Series) -> pd.Series:
    """Strip timezone + time-of-day from a Series index, returning a naive
    daily-resolution index. Safe to call on either tz-aware or naive series."""
    if s is None or len(s) == 0:
        return s
    out = s.copy()
    idx = pd.to_datetime(out.index)
    try:
        idx = idx.tz_convert(None)
    except (TypeError, AttributeError):
        try:
            idx = idx.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    out.index = pd.to_datetime(idx).normalize()
    # If multiple intra-day rows collapse onto the same date, keep the last.
    out = out[~out.index.duplicated(keep="last")]
    return out


def fmt_metric(v, kind="num"):
    """Format a metric value safely (handles NaN). kind: 'num', 'pct', 'ratio', 'days'."""
    if v is None or (isinstance(v, float) and (pd.isna(v) or not np.isfinite(v))):
        return "—"
    try:
        if kind == "pct":
            return f"{float(v) * 100:+.2f}%"
        if kind == "pct_unsigned":
            return f"{float(v) * 100:.2f}%"
        if kind == "ratio":
            return f"{float(v):.2f}"
        if kind == "days":
            return f"{int(v):,}"
        return f"{float(v):.3f}"
    except Exception:
        return "—"

def metric_card(label, value, delta=None, delta_sign=None):
    """delta_sign: 'pos', 'neg', 'neu', or None"""
    delta_html = ""
    if delta is not None:
        cls = f"delta-{delta_sign or 'neu'}"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def action_badge(action):
    color = ACTION_COLORS.get(str(action).lower(), "#8892a4")
    st.markdown(f"""
    <span class="action-badge" style="color:{color}; border-color:{color}40;">
        {action}
    </span>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 BR-PPO Trader")
    st.markdown("---")

    # ── Model selector (drives what the rest of the dashboard shows) ──
    registry = load_models_registry()
    model_ids   = [m["id"] for m in registry]
    model_names = {m["id"]: m.get("name", m["id"]) for m in registry}
    model_color = {m["id"]: m.get("color", "#4c9eff") for m in registry}

    default_idx = 0
    selected_id = st.selectbox(
        "Model",
        options=model_ids,
        index=default_idx,
        format_func=lambda mid: model_names.get(mid, mid),
        key="selected_model",
    )

    # Resolve the full model dict (carries repo/branch/logs_path)
    selected_model = next((m for m in registry if m["id"] == selected_id), registry[0])

    # Caption shows where the data is being fetched from
    _src_repo = selected_model.get("repo")
    _src_path = selected_model.get("logs_path") or f"logs/{selected_id}"
    if _src_repo:
        st.caption(f"Source: `{_src_repo}` / `{_src_path}`")
    else:
        st.caption(f"Showing: `{_src_path}/`")

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    data = load_all_for_model(selected_model)
    dec  = data["latest_decision"]
    port = data["portfolio"]

    # Last run time
    last_run = None
    if not dec.empty and "timestamp_utc" in dec.columns:
        last_run = pd.to_datetime(dec["timestamp_utc"].iloc[-1], utc=True, errors="coerce")
    if last_run is not None and pd.notna(last_run):
        now_utc  = datetime.now(timezone.utc)
        delta_s  = (now_utc - last_run).total_seconds()
        st.markdown(f"**Last run:** {last_run.strftime('%b %d, %H:%M UTC')}")
        if pd.notna(delta_s):
            delta_m = int(delta_s / 60)
            if delta_m < 60:
                st.markdown(f"*{delta_m} min ago*")
            elif delta_m < 1440:
                st.markdown(f"*{delta_m//60}h {delta_m%60}m ago*")
            else:
                st.markdown(f"*{delta_m//1440}d ago*")
    else:
        st.markdown("**Last run:** No data yet")

    st.markdown("---")

    # Account status
    if not dec.empty and "account_status" in dec.columns:
        status = str(dec["account_status"].iloc[-1])
        if status == "connected":
            st.markdown('<span class="status-live">● LIVE (Paper)</span>', unsafe_allow_html=True)
        elif status == "dry_run":
            st.markdown('<span class="status-dry">● DRY RUN</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-error">● UNKNOWN</span>', unsafe_allow_html=True)

    if not dec.empty and "submit_orders" in dec.columns:
        submitting = str(dec["submit_orders"].iloc[-1]).lower()
        st.markdown(f"**Order submission:** {'✅ Enabled' if submitting == 'true' else '⏸ Disabled'}")

    st.markdown("---")

    # Quick stats
    if not port.empty and "portfolio_value" in port.columns:
        latest_val = port["portfolio_value"].iloc[-1]
        st.markdown(f"**Portfolio:** {fmt_dollars(latest_val)}")

    if not dec.empty and "action" in dec.columns:
        latest_action = dec["action"].iloc[-1]
        st.markdown(f"**Current action:** `{latest_action}`")

    # Run count
    dec_hist = data["decisions"]
    if not dec_hist.empty:
        n_runs = len(dec_hist)
        st.markdown(f"**Total runs:** {n_runs}")

    st.markdown("---")
    st.caption("Auto-refreshes every 2 min\nLogs committed by GitHub Actions daily")


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 📈 BR-PPO Alpaca Paper Trading")
_subtitle = model_names.get(selected_id, selected_id)
st.markdown(f"*Viewing model: **{_subtitle}** (`{selected_id}`)*")
st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab_compare, tab_overview, tab_perf, tab_portfolio, tab_orders, tab_history, tab_health = st.tabs([
    "🔬 Compare Models",
    "🏠 Overview",
    "📊 Performance",
    "🎯 Portfolio",
    "📋 Orders",
    "📜 History",
    "🧠 Model Health",
])


# ════════════════════════════════════════════════════════════════
# TAB 0: COMPARE MODELS  (overlay equity curves across all enabled models)
# ════════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("### Side-by-side comparison of all enabled models")
    st.caption("Each model trades a separate Alpaca paper account. Curves below are normalized to 100 at first observation.")

    if len(registry) <= 1:
        st.info(
            "Only one model is currently registered. To enable comparison, add additional models to "
            "`models.yaml` and create their GitHub Environments. See README for the 5-step add-a-model recipe."
        )
    else:
        # Build per-model data dict
        all_data = {}
        for m in registry:
            mid = m["id"]
            all_data[mid] = {
                "name":      m.get("name", mid),
                "color":     m.get("color", "#4c9eff"),
                "model":     m,
                "portfolio": _normalize_portfolio_columns(load_model_csv(m, "portfolio/portfolio.csv")),
                "decisions": load_model_csv(m, "decisions/decisions.csv"),
                "health":    load_model_health(m),
            }

        # ── Equity curves overlay ──
        fig = go.Figure()
        any_data = False
        for mid, d in all_data.items():
            p = d["portfolio"]
            if p.empty or "portfolio_value" not in p.columns:
                continue
            ts_col = "timestamp_utc" if "timestamp_utc" in p.columns else None
            if ts_col is None:
                continue
            p = p.copy()
            p[ts_col] = pd.to_datetime(p[ts_col], utc=True, errors="coerce")
            p = p.dropna(subset=[ts_col]).sort_values(ts_col)
            if p.empty:
                continue
            base = float(p["portfolio_value"].iloc[0])
            if base <= 0:
                continue
            norm = p["portfolio_value"] / base * 100.0
            fig.add_trace(go.Scatter(
                x=p[ts_col], y=norm,
                mode="lines", name=d["name"],
                line=dict(color=d["color"], width=2.2),
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%b %d %Y}<br>Index: %{y:.2f}<extra></extra>",
            ))
            any_data = True

        # SPY benchmark for context
        try:
            import yfinance as yf
            first_dates = []
            for d in all_data.values():
                p = d["portfolio"]
                if not p.empty and "timestamp_utc" in p.columns:
                    fd = pd.to_datetime(p["timestamp_utc"], errors="coerce").min()
                    if pd.notna(fd):
                        first_dates.append(fd)
            if first_dates:
                start = min(first_dates).strftime("%Y-%m-%d")
                spy = yf.download("SPY", start=start, progress=False, auto_adjust=True)["Close"]
                if isinstance(spy, pd.DataFrame):
                    spy = spy.iloc[:, 0]
                spy = spy.dropna()
                if len(spy) > 1:
                    spy_norm = spy / float(spy.iloc[0]) * 100.0
                    fig.add_trace(go.Scatter(
                        x=spy.index, y=spy_norm,
                        mode="lines", name="SPY (benchmark)",
                        line=dict(color="#8892a4", width=1.5, dash="dash"),
                    ))
        except Exception:
            pass

        if not any_data:
            st.warning("No model has portfolio history yet. Compare view will populate after the first trade run for each model.")
        else:
            fig.update_layout(
                template="plotly_dark",
                height=480,
                title="Equity index — all models normalized to 100",
                xaxis_title=None, yaxis_title="Index (start = 100)",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Per-model summary table ──
        st.markdown("### Summary by model")
        summary_rows = []
        for mid, d in all_data.items():
            p = d["portfolio"]
            dec_h = d["decisions"]
            row = {"Model": d["name"], "ID": mid}
            if not p.empty and "portfolio_value" in p.columns:
                vals = p["portfolio_value"].dropna()
                if len(vals) >= 2:
                    rets = vals.pct_change().dropna()
                    total_ret = float(vals.iloc[-1] / vals.iloc[0] - 1.0)
                    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else None
                    row["Latest Value"] = fmt_dollars(float(vals.iloc[-1]))
                    row["Total Return"] = fmt_pct(total_ret)
                    row["Sharpe (ann.)"] = f"{sharpe:.2f}" if sharpe is not None else "—"
                else:
                    row["Latest Value"] = fmt_dollars(float(vals.iloc[-1])) if len(vals) else "—"
                    row["Total Return"] = "—"
                    row["Sharpe (ann.)"] = "—"
            else:
                row["Latest Value"] = "—"
                row["Total Return"] = "—"
                row["Sharpe (ann.)"] = "—"

            row["Runs"] = len(dec_h)
            icon, _, label, _ = _interpret_health(d.get("health") or {})
            row["Health"] = f"{icon} {label}"
            summary_rows.append(row)

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

        # ── Latest action per model ──
        st.markdown("### Latest action per model")
        action_cols = st.columns(min(len(all_data), 4) or 1)
        for i, (mid, d) in enumerate(all_data.items()):
            dec_latest = load_model_csv(d["model"], "decisions/latest_decision.csv")
            with action_cols[i % len(action_cols)]:
                if dec_latest.empty:
                    metric_card(d["name"], "no data")
                else:
                    a = str(dec_latest["action"].iloc[-1]) if "action" in dec_latest.columns else "—"
                    av = fmt_dollars(dec_latest["account_value"].iloc[-1]) if "account_value" in dec_latest.columns else ""
                    metric_card(d["name"], a, delta=av if av else None, delta_sign="neu")


# ════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════════════

with tab_overview:
    dec  = data["latest_decision"]
    port = data["portfolio"]

    if dec.empty:
        st.info("No trading data yet. The first run will populate this dashboard.")
        st.stop()

    latest = dec.iloc[-1]

    # ── Top KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = fmt_dollars(latest.get("account_value", 0))
        metric_card("Portfolio Value", val)

    with c2:
        action = str(latest.get("action", "—"))
        last   = str(latest.get("last_action", "—"))
        delta  = f"prev: {last}" if last != "—" and last != action else ("same as previous" if last == action and last != "—" else "first run")
        metric_card("Today's Action", action, delta=delta, delta_sign="neu")

    with c3:
        n_pos = int(latest.get("n_target_positions", 0))
        n_ord = int(latest.get("n_orders_planned", 0))
        metric_card("Target Positions", str(n_pos), delta=f"{n_ord} orders planned")

    with c4:
        n_sub = int(latest.get("n_orders_submitted", 0))
        submitted = str(latest.get("submit_orders", "False")).lower() == "true"
        sub_label = f"{n_sub} submitted" if submitted else "Dry run mode"
        sign = "pos" if n_sub > 0 else "neu"
        metric_card("Orders Submitted", str(n_sub), delta=sub_label, delta_sign=sign)

    st.markdown("---")

    # ── Decision detail + Portfolio mini-chart ──
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-header">Latest Decision</div>', unsafe_allow_html=True)
        decision_fields = [
            "market_date", "variant", "action", "last_action",
            "account_status", "account_value",
            "n_target_positions", "n_orders_planned", "n_orders_submitted",
        ]
        rows = []
        for f in decision_fields:
            if f in latest.index:
                v = latest[f]
                if f == "account_value":
                    v = fmt_dollars(v)
                rows.append({"Field": f.replace("_", " ").title(), "Value": str(v)})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Portfolio Value History</div>', unsafe_allow_html=True)
        if not port.empty and "portfolio_value" in port.columns:
            port_plot = port.copy()
            if "timestamp_utc" in port_plot.columns:
                port_plot["timestamp_utc"] = pd.to_datetime(port_plot["timestamp_utc"], utc=True)
                port_plot = port_plot.sort_values("timestamp_utc")
                x_col = "timestamp_utc"
            else:
                port_plot = port_plot.reset_index()
                x_col = "index"

            start_val = port_plot["portfolio_value"].iloc[0]
            end_val   = port_plot["portfolio_value"].iloc[-1]
            change    = end_val - start_val
            pct_chg   = (change / start_val * 100) if start_val else 0
            line_color = PALETTE["green"] if change >= 0 else PALETTE["red"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=port_plot[x_col],
                y=port_plot["portfolio_value"],
                mode="lines",
                line=dict(color=line_color, width=2.5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(line_color, 0.09),
                hovertemplate="$%{y:,.2f}<br>%{x}<extra></extra>",
            ))
            fig.update_layout(
                height=240,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color=PALETTE["muted"]),
                yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"],
                           tickprefix="$", tickformat=",.0f"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            sign = "+" if change >= 0 else ""
            color = PALETTE["green"] if change >= 0 else PALETTE["red"]
            st.markdown(f"<span style='color:{color}'>{sign}{fmt_dollars(change)} ({sign}{pct_chg:.2f}%) since first run</span>",
                        unsafe_allow_html=True)
        else:
            st.info("Portfolio history not yet available.")


# ════════════════════════════════════════════════════════════════
# TAB 2: PERFORMANCE
# ════════════════════════════════════════════════════════════════

with tab_perf:
    port      = data["portfolio"]
    decisions = data["decisions"]

    if port.empty:
        st.info("No portfolio history yet. Performance metrics populate after the first trading run.")
    else:
        # ── Prep daily-return series from portfolio_value ──
        port_plot = port.copy()
        if "timestamp_utc" in port_plot.columns:
            port_plot["timestamp_utc"] = pd.to_datetime(port_plot["timestamp_utc"], utc=True)
            port_plot = (port_plot.sort_values("timestamp_utc")
                                  .drop_duplicates("timestamp_utc"))
        port_plot["portfolio_value"] = pd.to_numeric(port_plot["portfolio_value"], errors="coerce")
        port_plot = port_plot.dropna(subset=["portfolio_value"])
        port_plot["daily_return"] = port_plot["portfolio_value"].pct_change()
        peak = port_plot["portfolio_value"].cummax()
        port_plot["drawdown"] = port_plot["portfolio_value"] / peak - 1

        # ── Period selector + benchmark toggle ──
        ctrl_l, ctrl_m, ctrl_r = st.columns([2, 2, 3])
        with ctrl_l:
            period_choice = st.selectbox(
                "Period",
                options=["All Time", "Last 30 Days", "Last 90 Days", "Year-to-Date"],
                index=0,
                key="perf_period",
            )
        with ctrl_m:
            bench_choice = st.selectbox(
                "Benchmark",
                options=["SPY", "QQQ", "RSP", "VTI", "None"],
                index=0,
                key="perf_bench",
            )

        # ── Slice by period ──
        x_vals_full = port_plot["timestamp_utc"]
        ts_max = x_vals_full.max()
        if period_choice == "Last 30 Days":
            cutoff = ts_max - pd.Timedelta(days=30)
            mask = x_vals_full >= cutoff
        elif period_choice == "Last 90 Days":
            cutoff = ts_max - pd.Timedelta(days=90)
            mask = x_vals_full >= cutoff
        elif period_choice == "Year-to-Date":
            cutoff = pd.Timestamp(ts_max.year, 1, 1, tz="UTC")
            mask = x_vals_full >= cutoff
        else:
            mask = pd.Series(True, index=port_plot.index)

        port_window = port_plot.loc[mask].copy()
        # Recompute drawdown within the window so the metric reflects the period
        if not port_window.empty:
            peak_w = port_window["portfolio_value"].cummax()
            port_window["drawdown"] = port_window["portfolio_value"] / peak_w - 1
            port_window["daily_return"] = port_window["portfolio_value"].pct_change()

        # ── Compute portfolio + benchmark statistics ──
        if not port_window.empty:
            port_returns = port_window["daily_return"].dropna()
            if len(port_returns) > 0:
                port_returns.index = pd.to_datetime(
                    port_window.loc[port_returns.index, "timestamp_utc"], utc=True
                )
        else:
            port_returns = pd.Series(dtype=float)

        port_stats = compute_perf_stats(port_returns)

        bench_stats = None
        bench_returns = pd.Series(dtype=float)
        bench_load_ok = False
        if bench_choice != "None" and not port_window.empty:
            start_ts = pd.to_datetime(port_window["timestamp_utc"], errors="coerce").min()
            if pd.notna(start_ts):
                start_str = start_ts.strftime("%Y-%m-%d")
                bench_returns = fetch_benchmark_returns(bench_choice, start=start_str)
            if not bench_returns.empty:
                bench_load_ok = True
                # Align on calendar date (drop time + tz from both sides)
                port_returns_d = _to_date_index(port_returns)
                bench_aligned = bench_returns.reindex(
                    port_returns_d.index, method="ffill"
                ).dropna()
                if not bench_aligned.empty:
                    bench_stats = compute_perf_stats(bench_aligned)

        # ── Performance metrics: 3 rows of KPIs ──
        st.markdown('<div class="section-header">Risk-Adjusted Performance</div>', unsafe_allow_html=True)

        # Row 1 — return / volatility
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            tr = port_stats["total_return"]
            metric_card(
                "Total Return",
                fmt_metric(tr, "pct"),
                delta=f"{port_stats['n_days']} obs",
                delta_sign="pos" if (tr or 0) >= 0 else "neg",
            )
        with r1c2:
            ar = port_stats["ann_return"]
            sub = (f"vs {bench_choice}: {fmt_metric(bench_stats['ann_return'], 'pct')}"
                   if bench_stats else "annualised")
            metric_card(
                "Ann. Return",
                fmt_metric(ar, "pct"),
                delta=sub,
                delta_sign="pos" if (ar or 0) >= 0 else "neg",
            )
        with r1c3:
            av = port_stats["ann_vol"]
            sub = (f"vs {bench_choice}: {fmt_metric(bench_stats['ann_vol'], 'pct_unsigned')}"
                   if bench_stats else "annualised")
            metric_card("Ann. Volatility", fmt_metric(av, "pct_unsigned"), delta=sub, delta_sign="neu")
        with r1c4:
            ts = port_stats["t_stat"]
            metric_card(
                "T-Stat",
                fmt_metric(ts, "ratio"),
                delta="|t|>2 is significant",
                delta_sign="pos" if (ts or 0) >= 2 else ("neg" if (ts or 0) <= -2 else "neu"),
            )

        # Row 2 — risk-adjusted ratios
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            sh = port_stats["sharpe"]
            sub = (f"vs {bench_choice}: {fmt_metric(bench_stats['sharpe'], 'ratio')}"
                   if bench_stats else "")
            sign = "pos" if (sh or 0) >= 1 else ("neu" if (sh or 0) >= 0 else "neg")
            metric_card("Sharpe Ratio", fmt_metric(sh, "ratio"), delta=sub or "annualised", delta_sign=sign)
        with r2c2:
            so = port_stats["sortino"]
            sub = (f"vs {bench_choice}: {fmt_metric(bench_stats['sortino'], 'ratio')}"
                   if bench_stats else "")
            sign = "pos" if (so or 0) >= 1 else ("neu" if (so or 0) >= 0 else "neg")
            metric_card("Sortino Ratio", fmt_metric(so, "ratio"), delta=sub or "downside-only", delta_sign=sign)
        with r2c3:
            ca = port_stats["calmar"]
            sub = (f"vs {bench_choice}: {fmt_metric(bench_stats['calmar'], 'ratio')}"
                   if bench_stats else "")
            sign = "pos" if (ca or 0) >= 1 else ("neu" if (ca or 0) >= 0 else "neg")
            metric_card("Calmar Ratio", fmt_metric(ca, "ratio"), delta=sub or "return / |maxDD|", delta_sign=sign)
        with r2c4:
            md = port_stats["max_dd"]
            sub = (f"vs {bench_choice}: {fmt_metric(bench_stats['max_dd'], 'pct_unsigned')}"
                   if bench_stats else f"{port_stats['longest_dd_days']} day max duration")
            metric_card("Max Drawdown", fmt_metric(md, "pct_unsigned"), delta=sub, delta_sign="neg")

        # Row 3 — win rate & trade economics
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        with r3c1:
            hr = port_stats["hit_rate"]
            wd, ld = port_stats["n_win_days"], port_stats["n_loss_days"]
            metric_card(
                "Hit Rate",
                fmt_metric(hr, "pct_unsigned"),
                delta=f"{wd} up days / {ld} down days",
                delta_sign="pos" if (hr or 0) >= 0.5 else "neg",
            )
        with r3c2:
            wlr = port_stats["win_loss_ratio"]
            avg_w = port_stats["avg_win"]
            avg_l = port_stats["avg_loss"]
            sub = (f"avg win {fmt_metric(avg_w, 'pct')} / loss {fmt_metric(avg_l, 'pct')}"
                   if not pd.isna(avg_w) and not pd.isna(avg_l) else "win vs loss size")
            metric_card("Win/Loss Ratio", fmt_metric(wlr, "ratio"), delta=sub,
                        delta_sign="pos" if (wlr or 0) >= 1 else "neg")
        with r3c3:
            pf = port_stats["profit_factor"]
            metric_card(
                "Profit Factor",
                fmt_metric(pf, "ratio"),
                delta="gross wins / gross losses",
                delta_sign="pos" if (pf or 0) >= 1.5 else ("neu" if (pf or 0) >= 1 else "neg"),
            )
        with r3c4:
            cdd = port_stats["current_dd"]
            cdd_days = port_stats["current_dd_days"]
            sub = f"{cdd_days} days off ATH" if cdd_days > 0 else "at all-time high"
            metric_card("Current Drawdown", fmt_metric(cdd, "pct_unsigned"), delta=sub,
                        delta_sign="neg" if (cdd or 0) < -0.001 else "pos")

        # Row 4 — extremes
        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        with r4c1:
            metric_card("Best Day", fmt_metric(port_stats["best_day"], "pct"), delta_sign="pos")
        with r4c2:
            metric_card("Worst Day", fmt_metric(port_stats["worst_day"], "pct"), delta_sign="neg")
        with r4c3:
            metric_card(
                "Longest Drawdown",
                f"{port_stats['longest_dd_days']} days",
                delta="consecutive below ATH",
                delta_sign="neu",
            )
        with r4c4:
            obs = port_stats["n_days"]
            metric_card("Observations", f"{obs:,}", delta=f"period: {period_choice}", delta_sign="neu")

        st.markdown("---")

        # ── Portfolio value chart with action annotations ──
        st.markdown('<div class="section-header">Portfolio Value</div>', unsafe_allow_html=True)
        fig = go.Figure()
        x_vals = port_window["timestamp_utc"] if not port_window.empty else port_plot["timestamp_utc"]
        y_vals = port_window["portfolio_value"] if not port_window.empty else port_plot["portfolio_value"]

        if len(y_vals) > 0:
            color = PALETTE["green"] if y_vals.iloc[-1] >= y_vals.iloc[0] else PALETTE["red"]
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                line=dict(color=color, width=2),
                fill="tozeroy", fillcolor=hex_to_rgba(color, 0.09),
                marker=dict(size=4, color=color),
                name="Portfolio",
                hovertemplate="$%{y:,.2f}<br>%{x}<extra></extra>",
            ))

            # Action annotations
            if not decisions.empty and "action" in decisions.columns and "timestamp_utc" in decisions.columns:
                decisions_plot = decisions.copy()
                decisions_plot["timestamp_utc"] = pd.to_datetime(decisions_plot["timestamp_utc"], utc=True)
                xmin, xmax = x_vals.min(), x_vals.max()
                decisions_plot = decisions_plot[
                    (decisions_plot["timestamp_utc"] >= xmin) &
                    (decisions_plot["timestamp_utc"] <= xmax)
                ]
                for _, row in decisions_plot.tail(30).iterrows():
                    fig.add_vline(
                        x=row["timestamp_utc"],
                        line_dash="dot",
                        line_color=ACTION_COLORS.get(str(row["action"]).lower(), "#8892a4"),
                        line_width=1,
                        opacity=0.5,
                    )

            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=16, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color=PALETTE["muted"]),
                yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"],
                           tickprefix="$", tickformat=",.0f"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Cumulative return vs benchmark ──
        if not port_returns.empty and bench_choice != "None":
            st.markdown(f'<div class="section-header">Cumulative Return vs {bench_choice}</div>',
                        unsafe_allow_html=True)

            if not bench_load_ok:
                st.info(f"Could not load {bench_choice} prices from yfinance. "
                        "Try refreshing in a minute, or pick a different benchmark.")
            else:
                # Both series on naive calendar-date index
                port_d  = _to_date_index(port_returns)
                bench_d = bench_returns.reindex(port_d.index, method="ffill").dropna()

                # Restrict portfolio side to the dates we actually have benchmark for,
                # so both lines start at the same x-value (zero).
                common_idx = port_d.index.intersection(bench_d.index)
                if len(common_idx) >= 2:
                    port_d  = port_d.loc[common_idx]
                    bench_d = bench_d.loc[common_idx]

                    port_cum  = (1 + port_d).cumprod()  - 1
                    bench_cum = (1 + bench_d).cumprod() - 1

                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=port_cum.index, y=port_cum.values * 100,
                        mode="lines+markers",
                        line=dict(color=PALETTE["blue"], width=2.5),
                        marker=dict(size=4),
                        name="Portfolio",
                        hovertemplate="%{y:.2f}%<extra>Portfolio</extra>",
                    ))
                    fig_cum.add_trace(go.Scatter(
                        x=bench_cum.index, y=bench_cum.values * 100,
                        mode="lines+markers",
                        line=dict(color=PALETTE["gold"], width=2, dash="dash"),
                        marker=dict(size=4),
                        name=bench_choice,
                        hovertemplate="%{y:.2f}%<extra>" + bench_choice + "</extra>",
                    ))
                    fig_cum.update_layout(
                        height=320,
                        margin=dict(l=0, r=0, t=8, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False, color=PALETTE["muted"]),
                        yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"], ticksuffix="%"),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["muted"]),
                                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                else:
                    st.info(f"Need at least 2 overlapping trading days between your portfolio and "
                            f"{bench_choice} to plot a comparison. You have {len(common_idx)} so far — "
                            "the chart will appear once you have a few more daily logs.")

        # ── Drawdown chart ──
        if "drawdown" in port_window.columns and len(port_window) > 1:
            st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=port_window["timestamp_utc"],
                y=port_window["drawdown"] * 100,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PALETTE["red"], width=1.5),
                fillcolor=hex_to_rgba(PALETTE["red"], 0.15),
                hovertemplate="%{y:.2f}%<br>%{x}<extra></extra>",
            ))
            fig2.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color=PALETTE["muted"]),
                yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"], ticksuffix="%"),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Rolling Sharpe (21-day window) ──
        if len(port_returns) >= 21:
            st.markdown('<div class="section-header">Rolling 21-Day Sharpe Ratio</div>',
                        unsafe_allow_html=True)
            roll_mean = port_returns.rolling(21).mean()
            roll_std  = port_returns.rolling(21).std()
            roll_sharpe = (roll_mean / roll_std * np.sqrt(252)).dropna()

            fig_rs = go.Figure()
            fig_rs.add_hline(y=0, line_dash="solid", line_color=PALETTE["muted"], line_width=1)
            fig_rs.add_hline(y=1, line_dash="dot", line_color=PALETTE["green"], opacity=0.5,
                             annotation_text="Sharpe = 1", annotation_position="right")
            fig_rs.add_trace(go.Scatter(
                x=roll_sharpe.index, y=roll_sharpe.values,
                mode="lines",
                line=dict(color=PALETTE["blue"], width=2),
                fill="tozeroy",
                fillcolor=hex_to_rgba(PALETTE["blue"], 0.10),
                hovertemplate="%{y:.2f}<br>%{x}<extra></extra>",
            ))
            fig_rs.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color=PALETTE["muted"]),
                yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"]),
                showlegend=False,
            )
            st.plotly_chart(fig_rs, use_container_width=True)

        # ── Monthly returns heatmap ──
        if len(port_returns) >= 20:
            st.markdown('<div class="section-header">Monthly Returns Heatmap</div>',
                        unsafe_allow_html=True)
            # "ME" (month-end) is the modern label; fall back to "M" on older pandas.
            try:
                monthly = (1 + port_returns).resample("ME").prod() - 1
            except (ValueError, KeyError):
                monthly = (1 + port_returns).resample("M").prod() - 1
            if len(monthly) >= 1:
                df_m = pd.DataFrame({
                    "year":  monthly.index.year,
                    "month": monthly.index.month,
                    "ret":   monthly.values * 100,
                })
                pivot = df_m.pivot(index="year", columns="month", values="ret")
                # Ensure all 12 columns exist
                for m in range(1, 13):
                    if m not in pivot.columns:
                        pivot[m] = np.nan
                pivot = pivot[sorted(pivot.columns)]
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

                # Text labels (NaN → blank). Use .map for pandas>=2.1, fall back to .applymap.
                _fmt_cell = lambda v: "" if pd.isna(v) else f"{v:+.1f}%"
                try:
                    text_vals = pivot.map(_fmt_cell)
                except AttributeError:
                    text_vals = pivot.applymap(_fmt_cell)

                fig_hm = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=month_names,
                    y=[str(y) for y in pivot.index],
                    text=text_vals.values,
                    texttemplate="%{text}",
                    textfont=dict(color="#e8eaf0", size=11),
                    colorscale=[
                        [0.0, PALETTE["red"]],
                        [0.5, "#1a1f2e"],
                        [1.0, PALETTE["green"]],
                    ],
                    zmid=0,
                    hovertemplate="%{y} %{x}: %{z:+.2f}%<extra></extra>",
                    colorbar=dict(title="%", tickfont=dict(color=PALETTE["muted"])),
                ))
                fig_hm.update_layout(
                    height=max(180, 60 + 32 * len(pivot.index)),
                    margin=dict(l=0, r=0, t=8, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color=PALETTE["muted"], side="top"),
                    yaxis=dict(color=PALETTE["muted"]),
                )
                st.plotly_chart(fig_hm, use_container_width=True)

    # ── Action distribution (kept as-is, useful at-a-glance) ──
    if not decisions.empty and "action" in decisions.columns:
        st.markdown('<div class="section-header">Action Distribution (All Time)</div>',
                    unsafe_allow_html=True)
        action_counts = decisions["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        colors = [ACTION_COLORS.get(a, "#8892a4") for a in action_counts["action"]]

        fig3 = go.Figure(go.Bar(
            x=action_counts["action"],
            y=action_counts["count"],
            marker_color=colors,
            hovertemplate="%{x}: %{y} times<extra></extra>",
        ))
        fig3.update_layout(
            height=240,
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color=PALETTE["muted"]),
            yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"]),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Footnote on metric definitions ──
    with st.expander("ℹ️ Metric definitions"):
        st.markdown("""
- **Total Return** — cumulative return over the selected period.
- **Annualised Return** — geometric (CAGR-style) annualisation: `(1 + total)^(252/n) − 1`.
- **Annualised Volatility** — daily return std × √252.
- **Sharpe Ratio** — daily mean / daily std × √252 (rf assumed 0). Above 1 is generally considered good for an equity strategy.
- **Sortino Ratio** — uses only the std of negative returns in the denominator. Penalises downside only.
- **Calmar Ratio** — annualised return divided by absolute max drawdown. Captures return per unit of worst-case pain.
- **Max Drawdown** — largest peak-to-trough decline within the period.
- **Hit Rate** — share of days with positive returns.
- **Win/Loss Ratio** — average up-day return divided by absolute average down-day return. >1 means winners outsize losers on average.
- **Profit Factor** — sum of all positive returns divided by absolute sum of all negative returns.
- **T-Stat** — `mean / std × √n` on daily returns. Rough significance test that returns ≠ 0; |t| > 2 is conventional.
- **Longest Drawdown** — most consecutive trading days spent below the running peak.
- **Current Drawdown** — distance from the most recent all-time high in the period.

Note: these metrics are computed off the daily portfolio-value series logged after each run, so reliability improves once you have several weeks of paper trading data.
""")


# ════════════════════════════════════════════════════════════════
# TAB 3: PORTFOLIO
# ════════════════════════════════════════════════════════════════

with tab_portfolio:
    tw  = data["target_weights"]
    pos = data["positions"]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Target Weights</div>', unsafe_allow_html=True)

        if tw.empty:
            st.info("No target weights yet.")
        else:
            tw_plot = tw.copy()
            if "target_weight" in tw_plot.columns and "symbol" in tw_plot.columns:
                tw_plot = tw_plot.sort_values("target_weight", ascending=False)

                # Pie chart
                fig = go.Figure(go.Pie(
                    labels=tw_plot["symbol"],
                    values=tw_plot["target_weight"],
                    hole=0.45,
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value:.2%}<extra></extra>",
                    marker=dict(
                        colors=px.colors.qualitative.Plotly + px.colors.qualitative.Pastel,
                        line=dict(color="#0e1117", width=2),
                    ),
                ))
                fig.update_layout(
                    height=320,
                    margin=dict(l=0, r=0, t=8, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color=PALETTE["muted"]), bgcolor="rgba(0,0,0,0)"),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table
                tw_display = tw_plot[["symbol", "target_weight"]].copy()
                tw_display["target_weight"] = tw_display["target_weight"].apply(lambda x: f"{x:.2%}")
                st.dataframe(tw_display, hide_index=True, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Current Positions (Alpaca)</div>', unsafe_allow_html=True)

        if pos.empty:
            st.info("No position data. Positions are populated when Alpaca is connected.")
        else:
            # Bar chart
            if "symbol" in pos.columns and "weight" in pos.columns:
                pos_plot = pos.sort_values("weight", ascending=False)
                fig2 = go.Figure(go.Bar(
                    x=pos_plot["symbol"],
                    y=pos_plot["weight"],
                    marker_color=PALETTE["blue"],
                    text=[f"{v:.1%}" for v in pos_plot["weight"]],
                    textposition="outside",
                    hovertemplate="%{x}: %{y:.2%}<br>$%{customdata:,.2f}<extra></extra>",
                    customdata=pos_plot.get("market_value", pd.Series([0]*len(pos_plot))),
                ))
                fig2.update_layout(
                    height=320,
                    margin=dict(l=0, r=0, t=8, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, color=PALETTE["muted"]),
                    yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"],
                               tickformat=".0%"),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Position table
            pos_display = pos.copy()
            if "market_value" in pos_display.columns:
                pos_display["market_value"] = pos_display["market_value"].apply(fmt_dollars)
            if "weight" in pos_display.columns:
                pos_display["weight"] = pos_display["weight"].apply(lambda x: f"{x:.2%}")
            st.dataframe(pos_display, hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4: ORDERS
# ════════════════════════════════════════════════════════════════

with tab_orders:
    planned   = data["planned_orders"]
    submitted = data["submitted_orders"]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Planned Orders (Latest Run)</div>', unsafe_allow_html=True)
        if planned.empty:
            st.info("No planned orders from the latest run.")
        else:
            disp = planned.copy()
            for col in ["current_weight", "target_weight", "delta_weight"]:
                if col in disp.columns:
                    disp[col] = disp[col].apply(lambda x: f"{float(x):.2%}" if pd.notna(x) else "—")
            if "notional" in disp.columns:
                disp["notional"] = disp["notional"].apply(lambda x: fmt_dollars(x))

            # Color code side
            def style_side(row):
                if row.get("side") == "buy":
                    return ["color: #00d4aa"] * len(row)
                elif row.get("side") == "sell":
                    return ["color: #ff4b6e"] * len(row)
                return [""] * len(row)

            st.dataframe(disp, hide_index=True, use_container_width=True)
            total_notional = planned["notional"].sum() if "notional" in planned.columns else 0
            st.caption(f"Total notional: {fmt_dollars(total_notional)}")

    with col_r:
        st.markdown('<div class="section-header">Submitted Orders (Latest Run)</div>', unsafe_allow_html=True)
        if submitted.empty:
            st.info("No submitted orders yet (may be dry-run mode).")
        else:
            disp = submitted.copy()
            for col in ["current_weight", "target_weight", "delta_weight"]:
                if col in disp.columns:
                    disp[col] = disp[col].apply(lambda x: f"{float(x):.2%}" if pd.notna(x) else "—")
            if "notional" in disp.columns:
                disp["notional"] = disp["notional"].apply(lambda x: fmt_dollars(x))
            st.dataframe(disp, hide_index=True, use_container_width=True)

            if "submitted" in submitted.columns:
                n_ok  = int(submitted["submitted"].sum())
                n_err = len(submitted) - n_ok
                if n_err > 0:
                    st.warning(f"⚠️ {n_err} order(s) failed to submit.")
                else:
                    st.success(f"✓ All {n_ok} orders submitted successfully.")


# ════════════════════════════════════════════════════════════════
# TAB 5: HISTORY
# ════════════════════════════════════════════════════════════════

with tab_history:
    decisions     = data["decisions"]
    orders_hist   = data["orders_history"]

    st.markdown('<div class="section-header">All Decisions</div>', unsafe_allow_html=True)

    if decisions.empty:
        st.info("No decision history yet.")
    else:
        dec_display = decisions.copy()
        if "account_value" in dec_display.columns:
            dec_display["account_value"] = dec_display["account_value"].apply(fmt_dollars)
        if "timestamp_utc" in dec_display.columns:
            dec_display["timestamp_utc"] = dec_display["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M UTC")

        cols_show = [c for c in [
            "timestamp_utc", "market_date", "action", "last_action",
            "account_value", "n_target_positions", "n_orders_planned",
            "n_orders_submitted", "account_status", "submit_orders",
        ] if c in dec_display.columns]

        st.dataframe(
            dec_display[cols_show].sort_values("timestamp_utc", ascending=False)
                       .reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-header">All Submitted Orders</div>', unsafe_allow_html=True)

    if orders_hist.empty:
        st.info("No order history yet.")
    else:
        ord_display = orders_hist.copy()
        if "timestamp_utc" in ord_display.columns:
            ord_display["timestamp_utc"] = pd.to_datetime(ord_display["timestamp_utc"], utc=True)
            ord_display["timestamp_utc"] = ord_display["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M UTC")
        for col in ["current_weight", "target_weight", "delta_weight"]:
            if col in ord_display.columns:
                ord_display[col] = ord_display[col].apply(
                    lambda x: f"{float(x):.2%}" if pd.notna(x) else "—")
        if "notional" in ord_display.columns:
            ord_display["notional"] = ord_display["notional"].apply(fmt_dollars)

        st.dataframe(
            ord_display.sort_values("timestamp_utc", ascending=False).reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════
# TAB 6: MODEL HEALTH
# ════════════════════════════════════════════════════════════════

with tab_health:
    health   = data.get("health_status", {})
    sig_hist = data.get("signal_history", pd.DataFrame())
    dec_all  = data.get("decisions", pd.DataFrame())
    port_all = data.get("portfolio", pd.DataFrame())

    # ── Header status banner ──
    icon, sc, label, is_pending = _interpret_health(health or {})
    last_checked = (health or {}).get("computed_at", "")
    last_checked_str = last_checked[:19].replace("T", " ") if last_checked else "Never"
    n_dec = (health or {}).get("n_decisions") or 0
    subline = (
        f"Awaiting more trading data ({n_dec} of 5 runs required for first check)"
        if is_pending else
        f"Last checked: {last_checked_str} UTC  |  "
        f"Lookback: {(health or {}).get('lookback_days', 63)} days"
    )

    st.markdown(f"""
    <div style="background:{sc}22; border:2px solid {sc}; border-radius:12px; padding:20px 24px; margin-bottom:20px;">
        <span style="font-size:24px; font-weight:700; color:{sc};">{icon} Model Status: {label}</span>
        <div style="color:#8892a4; font-size:13px; margin-top:6px;">{subline}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Alerts ──
    alerts = health.get("alerts", [])
    if alerts:
        for alert in alerts:
            st.warning(alert)
    elif label == "Healthy":
        st.success("No alerts. The model is performing as expected.")

    if health.get("training_recommended"):
        st.error("🔁 **Retraining recommended.** See the 'Retrain' section below for instructions.")

    st.markdown("---")

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        ent = health.get("action_entropy")
        ent_str = f"{ent:.3f}" if ent is not None else "—"
        color = "#00d4aa" if (ent or 0) > 0.5 else ("#ffd166" if (ent or 0) > 0.3 else "#ff4b6e")
        metric_card("Action Entropy", ent_str, "1.0 = fully diverse, 0 = locked in", "pos" if (ent or 0) > 0.5 else "neg")

    with c2:
        n_unique = health.get("n_unique_actions", 0)
        n_dec    = health.get("n_decisions", 0)
        metric_card("Unique Actions Used", str(n_unique), f"out of last {n_dec} decisions", "pos" if n_unique > 3 else "neg")

    with c3:
        p_sh = health.get("portfolio_sharpe_30d")
        b_sh = health.get("spy_sharpe_30d")
        if p_sh is not None and b_sh is not None:
            gap = p_sh - b_sh
            metric_card("Sharpe vs SPY (30d)", f"{p_sh:.2f}", f"SPY: {b_sh:.2f} | gap: {gap:+.2f}", "pos" if gap > -0.3 else "neg")
        else:
            metric_card("Sharpe vs SPY (30d)", "—", "Insufficient data", "neu")

    with c4:
        p_ret = health.get("portfolio_return_30d")
        b_ret = health.get("spy_return_30d")
        if p_ret is not None and b_ret is not None:
            gap = p_ret - b_ret
            metric_card("Return vs SPY (30d)", f"{p_ret*100:.1f}%", f"SPY: {b_ret*100:.1f}% | gap: {gap*100:+.1f}%", "pos" if gap > -0.05 else "neg")
        else:
            metric_card("Return vs SPY (30d)", "—", "Insufficient data", "neu")

    st.markdown("---")

    # ── Action distribution ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Action Distribution (All Time)</div>', unsafe_allow_html=True)
        action_counts = health.get("action_counts", {})

        if action_counts and not dec_all.empty and "action" in dec_all.columns:
            all_counts = dec_all["action"].value_counts().to_dict()
            fig_pie = go.Figure(go.Pie(
                labels=list(all_counts.keys()),
                values=list(all_counts.values()),
                hole=0.45,
                marker=dict(colors=[ACTION_COLORS.get(a, "#8892a4") for a in all_counts.keys()]),
                textinfo="label+percent",
                textfont=dict(size=12, color="#e8eaf0"),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eaf0"),
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No action data yet.")

    with col_right:
        st.markdown('<div class="section-header">Action Over Time</div>', unsafe_allow_html=True)
        if not dec_all.empty and "action" in dec_all.columns and "timestamp_utc" in dec_all.columns:
            dec_plot = dec_all.copy()
            dec_plot["timestamp_utc"] = pd.to_datetime(dec_plot["timestamp_utc"], utc=True)
            dec_plot = dec_plot.sort_values("timestamp_utc").tail(120)

            fig_act = go.Figure()
            for action_name in dec_plot["action"].unique():
                mask = dec_plot["action"] == action_name
                col  = ACTION_COLORS.get(action_name, "#8892a4")
                fig_act.add_trace(go.Scatter(
                    x=dec_plot.loc[mask, "timestamp_utc"],
                    y=dec_plot.loc[mask, "action"],
                    mode="markers",
                    name=action_name,
                    marker=dict(color=col, size=10, symbol="circle"),
                ))
            fig_act.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#12161f",
                font=dict(color="#e8eaf0"),
                xaxis=dict(gridcolor="#1e2535", title="Date"),
                yaxis=dict(gridcolor="#1e2535", title="Action", automargin=True),
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
            )
            st.plotly_chart(fig_act, use_container_width=True)
        else:
            st.info("No action history yet.")

    st.markdown("---")

    # ── Rolling entropy trend ──
    st.markdown('<div class="section-header">Rolling Action Entropy (21-day window)</div>', unsafe_allow_html=True)
    if not dec_all.empty and "action" in dec_all.columns and "timestamp_utc" in dec_all.columns:
        dec_sorted = dec_all.copy()
        dec_sorted["timestamp_utc"] = pd.to_datetime(dec_sorted["timestamp_utc"], utc=True)
        dec_sorted = dec_sorted.sort_values("timestamp_utc").reset_index(drop=True)

        window = 21
        entropy_vals = []
        for i in range(len(dec_sorted)):
            start = max(0, i - window + 1)
            window_actions = dec_sorted["action"].iloc[start:i+1]
            counts = window_actions.value_counts().values
            p = counts / counts.sum()
            H = -float(np.sum(p * np.log(p + 1e-12)))
            H_max = math.log(max(len(counts), 2))
            entropy_vals.append(H / H_max if H_max > 0 else 0.0)

        dec_sorted["entropy_21d"] = entropy_vals

        fig_ent = go.Figure()
        fig_ent.add_hrect(y0=0, y1=0.3,  fillcolor="rgba(255,75,110,0.06)",   line_width=0)
        fig_ent.add_hrect(y0=0.3, y1=0.6, fillcolor="rgba(255,209,102,0.06)", line_width=0)
        fig_ent.add_hrect(y0=0.6, y1=1.0, fillcolor="rgba(0,212,170,0.06)",   line_width=0)
        fig_ent.add_hline(y=0.3, line_dash="dot", line_color="#ff4b6e", annotation_text="Degraded", annotation_position="right")
        fig_ent.add_hline(y=0.6, line_dash="dot", line_color="#ffd166", annotation_text="Warning",  annotation_position="right")
        fig_ent.add_trace(go.Scatter(
            x=dec_sorted["timestamp_utc"],
            y=dec_sorted["entropy_21d"],
            mode="lines",
            line=dict(color="#4c9eff", width=2),
            name="Entropy",
        ))
        fig_ent.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#12161f",
            font=dict(color="#e8eaf0"),
            xaxis=dict(gridcolor="#1e2535"),
            yaxis=dict(gridcolor="#1e2535", range=[0, 1], title="Entropy (0–1)"),
            margin=dict(t=10, b=10, l=10, r=40),
            height=220,
            showlegend=False,
        )
        st.plotly_chart(fig_ent, use_container_width=True)
    else:
        st.info("Need decision history to compute rolling entropy.")

    # ── Portfolio vs SPY ──
    st.markdown("---")
    st.markdown('<div class="section-header">Portfolio Value vs SPY (normalised)</div>', unsafe_allow_html=True)

    if not port_all.empty and "portfolio_value" in port_all.columns and "timestamp_utc" in port_all.columns:
        port_plot = port_all.copy()
        port_plot["timestamp_utc"] = pd.to_datetime(port_plot["timestamp_utc"], utc=True)
        port_plot = port_plot.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
        port_plot = port_plot.set_index("timestamp_utc")["portfolio_value"].astype(float)
        port_norm = port_plot / port_plot.iloc[0]

        try:
            import yfinance as yf
            start_idx = port_plot.index[0] if len(port_plot) else None
            if start_idx is None or pd.isna(start_idx):
                raise ValueError("no valid start date for SPY benchmark")
            spy_data = yf.download("SPY", start=start_idx.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
            spy_close = spy_data["Close"] if "Close" in spy_data.columns else spy_data.iloc[:, 0]
            spy_close.index = pd.to_datetime(spy_close.index, utc=True)
            spy_norm = spy_close / spy_close.iloc[0]
            has_spy = True
        except Exception:
            has_spy = False

        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=port_norm.index, y=port_norm.values,
            name="Portfolio", line=dict(color="#4c9eff", width=2),
        ))
        if has_spy:
            spy_reindexed = spy_norm.reindex(port_norm.index, method="ffill")
            fig_pv.add_trace(go.Scatter(
                x=spy_reindexed.index, y=spy_reindexed.values,
                name="SPY", line=dict(color="#ffd166", width=2, dash="dash"),
            ))
        fig_pv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#12161f",
            font=dict(color="#e8eaf0"),
            xaxis=dict(gridcolor="#1e2535"),
            yaxis=dict(gridcolor="#1e2535", title="Normalised value"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0")),
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_pv, use_container_width=True)
    else:
        st.info("No portfolio value history yet.")

    # ── Retraining guide ──
    st.markdown("---")
    with st.expander("🔁 When and how to retrain", expanded=health.get("training_recommended", False)):
        st.markdown("""
**Retrain when you see any of these:**
- 🚨 Model status is **DEGRADED** (action lock-in or sustained underperformance)
- ⚠️ Action entropy stays below 0.3 for several weeks
- ⚠️ Portfolio trails SPY by >15% over a 30-day period
- 📅 **Quarterly schedule** (Jan, Apr, Jul, Oct) — regardless of metrics

**How to retrain:**
1. Open the Colab notebook: `vision_ichimoku_agentic.py`
2. Run **all cells through V10B** (the BR-PPO tuning section)
3. The trained model is saved as `v10_bro_ppo_allocation_agent.zip`
4. Download it and replace `artifacts/v10_bro_ppo_allocation_agent.zip` in this repo
5. Also replace `artifacts/v10_bro_ppo_allocation_agent_metadata.json`
6. Commit and push — the daily workflow picks up the new model automatically

> **GPU tip:** The vision embeddings (V6 cells) need a T4 GPU. Use Google Colab Pro or a cloud GPU instance.
> The PPO retraining alone (V10B cells only) can run on CPU in ~30 minutes if your cached alpha scores exist.
""")

    st.markdown("---")
    st.caption(f"Health check runs weekly (Fridays) and quarterly. Last computed: {health.get('computed_at', 'Never')[:19].replace('T', ' ')} UTC")

