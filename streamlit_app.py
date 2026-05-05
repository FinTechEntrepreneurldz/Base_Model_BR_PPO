"""
QSentia BR-PPO Investor Research Terminal
Streamlit Cloud dashboard for Alpaca paper trading logs.

UI-only redesign.
Backend assumptions preserved:
- models.yaml model registry
- logs/<model_id>/... CSV structure
- optional multi-repo raw.githubusercontent.com model log loading
- committed paper trading logs from GitHub Actions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timezone
import json


# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="QSentia BR-PPO Research Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_ROOT = Path(__file__).parent
LOGS_ROOT = REPO_ROOT / "logs"
MODELS_YAML = REPO_ROOT / "models.yaml"
RAW_BASE = "https://raw.githubusercontent.com"


# =============================================================================
# DESIGN SYSTEM
# =============================================================================

PALETTE = {
    "bg": "#080b12",
    "panel": "#101522",
    "panel_2": "#151b2b",
    "border": "rgba(255,255,255,0.10)",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "soft": "#cbd5e1",
    "green": "#00d4aa",
    "red": "#ff4b6e",
    "blue": "#4c9eff",
    "gold": "#ffd166",
    "purple": "#a78bfa",
    "orange": "#f97316",
}

ACTION_COLORS = {
    "current_ew": PALETTE["blue"],
    "top_ew": PALETTE["green"],
    "spy": PALETTE["gold"],
    "v6_alpha": PALETTE["purple"],
    "v8_blend": PALETTE["orange"],
    "bil_cash": PALETTE["muted"],
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(76,158,255,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(0,212,170,0.10), transparent 32%),
        linear-gradient(180deg, #080b12 0%, #0e1117 48%, #080b12 100%);
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: rgba(8,11,18,0.70);
    backdrop-filter: blur(16px);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f18 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #f8fafc !important;
    letter-spacing: -0.04em !important;
}

p, span, div, label {
    color: #dbe4f0;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent);
    margin: 1.1rem 0 1.35rem 0;
}

.hero {
    background:
        linear-gradient(135deg, rgba(18,24,38,0.98), rgba(9,13,22,0.98)),
        radial-gradient(circle at top right, rgba(0,212,170,0.16), transparent 38%);
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 26px;
    padding: 30px 34px;
    margin-bottom: 22px;
    box-shadow: 0 24px 90px rgba(0,0,0,0.42);
}

.hero-kicker {
    color: #00d4aa;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero-title {
    color: #f8fafc;
    font-size: 38px;
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -0.055em;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #aab7cc;
    font-size: 15px;
    line-height: 1.62;
    max-width: 1050px;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 20px;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.11);
    color: #dce7f7;
    font-size: 12px;
    font-weight: 700;
}

.card {
    background: linear-gradient(180deg, rgba(21,27,43,0.97), rgba(13,18,30,0.97));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 18px 55px rgba(0,0,0,0.26);
}

.metric-card {
    background: linear-gradient(180deg, rgba(21,27,43,0.98), rgba(13,18,30,0.98));
    border: 1px solid rgba(255,255,255,0.095);
    border-radius: 20px;
    padding: 20px 22px;
    min-height: 124px;
    margin-bottom: 14px;
    box-shadow: 0 18px 55px rgba(0,0,0,0.26);
}

.metric-card:hover {
    border-color: rgba(76,158,255,0.35);
    transform: translateY(-1px);
    transition: all 0.16s ease;
}

.metric-label {
    color: #8f9db2;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 9px;
}

.metric-value {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: -0.045em;
    line-height: 1.08;
}

.metric-delta {
    font-size: 13px;
    font-weight: 700;
    margin-top: 8px;
}

.delta-pos { color: #00d4aa; }
.delta-neg { color: #ff4b6e; }
.delta-neu { color: #9aa6bb; }

.section-title {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 900;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.55;
    margin-bottom: 16px;
}

.action-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: rgba(76,158,255,0.10);
    color: #7bb6ff;
    border: 1px solid rgba(76,158,255,0.35);
}

.status-live, .status-dry, .status-error, .status-pending {
    display: inline-flex;
    align-items: center;
    padding: 7px 11px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 900;
}

.status-live {
    color: #00d4aa;
    background: rgba(0,212,170,0.10);
    border: 1px solid rgba(0,212,170,0.28);
}

.status-dry {
    color: #ffd166;
    background: rgba(255,209,102,0.10);
    border: 1px solid rgba(255,209,102,0.28);
}

.status-error {
    color: #ff4b6e;
    background: rgba(255,75,110,0.10);
    border: 1px solid rgba(255,75,110,0.28);
}

.status-pending {
    color: #9aa6bb;
    background: rgba(148,163,184,0.10);
    border: 1px solid rgba(148,163,184,0.24);
}

.sidebar-brand {
    font-size: 25px;
    font-weight: 900;
    color: #f8fafc;
    letter-spacing: -0.05em;
    margin-bottom: 0;
}

.sidebar-subtitle {
    color: #00d4aa;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 18px;
}

.sidebar-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px 15px;
    margin: 12px 0;
}

.sidebar-label {
    color: #8f9db2;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.sidebar-value {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 800;
    line-height: 1.4;
}

.disclaimer {
    color: #8f9db2;
    font-size: 12px;
    line-height: 1.55;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(18,22,31,0.88);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 6px;
    gap: 4px;
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab"] {
    color: #9aa6bb;
    border-radius: 13px;
    padding: 10px 14px;
    font-weight: 800;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(76,158,255,0.20), rgba(0,212,170,0.12)) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255,255,255,0.10);
}

div[data-testid="stPlotlyChart"] {
    background: rgba(16,21,33,0.72);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 10px;
    box-shadow: 0 18px 55px rgba(0,0,0,0.20);
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #00a884);
    color: white;
    border: none;
    border-radius: 13px;
    font-weight: 900;
    padding: 0.65rem 1rem;
    box-shadow: 0 12px 35px rgba(31,111,235,0.22);
}

.stButton > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
    transition: all 0.15s ease;
}

.stSelectbox label {
    color: #aab4c8 !important;
    font-weight: 800 !important;
}

.small-muted {
    color: #8f9db2;
    font-size: 12px;
    line-height: 1.5;
}

.footer-note {
    color: #7f8ca3;
    font-size: 11px;
    line-height: 1.5;
    margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=60)
def load_models_registry():
    """Read models.yaml; return enabled models. Fallback to logs folders."""
    try:
        import yaml
        with open(MODELS_YAML) as f:
            data = yaml.safe_load(f) or {}
        enabled = [m for m in (data.get("models") or []) if m.get("enabled", True)]
        if enabled:
            return enabled
    except Exception:
        pass

    fallback = []
    if LOGS_ROOT.exists():
        for sub in sorted(LOGS_ROOT.iterdir()):
            if sub.is_dir() and (sub / "decisions").exists():
                fallback.append(
                    {
                        "id": sub.name,
                        "name": sub.name,
                        "color": PALETTE["blue"],
                        "environment": sub.name,
                    }
                )

    if fallback:
        return fallback

    return [
        {
            "id": "default",
            "name": "Default",
            "color": PALETTE["blue"],
            "environment": "default",
        }
    ]


@st.cache_data(ttl=120)
def load_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _load_health_status(log_dir: Path):
    path = Path(log_dir) / "health" / "health_status.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_all(log_dir: Path):
    log_dir = Path(log_dir)
    return {
        "decisions": load_csv(log_dir / "decisions" / "decisions.csv"),
        "latest_decision": load_csv(log_dir / "decisions" / "latest_decision.csv"),
        "portfolio": load_csv(log_dir / "portfolio" / "portfolio.csv"),
        "target_weights": load_csv(log_dir / "target_weights" / "latest_target_weights.csv"),
        "tw_history": load_csv(log_dir / "target_weights" / "target_weights.csv"),
        "positions": load_csv(log_dir / "positions" / "latest_positions.csv"),
        "planned_orders": load_csv(log_dir / "orders" / "latest_planned_orders.csv"),
        "submitted_orders": load_csv(log_dir / "orders" / "latest_submitted_orders.csv"),
        "orders_history": load_csv(log_dir / "orders" / "submitted_orders.csv"),
        "signal_history": load_csv(log_dir / "health" / "signal_history.csv"),
        "health_status": _load_health_status(log_dir),
    }


def _model_logs_url(model: dict, relpath: str) -> str | None:
    repo = model.get("repo")
    if not repo:
        return None
    branch = model.get("branch", "main")
    logs_path = (model.get("logs_path") or f"logs/{model['id']}").strip("/")
    relpath = relpath.strip("/")
    return f"{RAW_BASE}/{repo}/{branch}/{logs_path}/{relpath}"


def _parse_timestamp_column(s):
    s2 = s.astype(str).str.strip().str.replace("_", "T", regex=False)
    return pd.to_datetime(s2, utc=True, errors="coerce")


@st.cache_data(ttl=120, show_spinner=False)
def _load_csv_url(url: str) -> pd.DataFrame:
    try:
        import io
        import requests

        headers = {
            "User-Agent": "qsentia-research-terminal/1.0",
            "Cache-Control": "no-cache",
        }
        r = requests.get(url, timeout=15, headers=headers)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = _parse_timestamp_column(df["timestamp_utc"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def _load_json_url(url: str) -> dict:
    try:
        import requests

        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def load_model_csv(model: dict, relpath: str) -> pd.DataFrame:
    url = _model_logs_url(model, relpath)
    if url is not None:
        return _load_csv_url(url)

    logs_path = model.get("logs_path") or f"logs/{model['id']}"
    return load_csv(REPO_ROOT / logs_path / relpath)


def load_model_health(model: dict) -> dict:
    central_path = REPO_ROOT / "logs" / model["id"] / "health" / "health_status.json"
    if central_path.exists():
        return _load_health_status(central_path.parent.parent)
    return {}


def _normalize_portfolio_columns(df):
    if df is None or not hasattr(df, "columns") or df.empty:
        return df
    if "portfolio_value" in df.columns:
        return df
    for alt in ("equity", "account_value", "total_equity"):
        if alt in df.columns:
            df = df.copy()
            df["portfolio_value"] = pd.to_numeric(df[alt], errors="coerce")
            return df
    return df


def load_all_for_model(model: dict) -> dict:
    return {
        "decisions": load_model_csv(model, "decisions/decisions.csv"),
        "latest_decision": load_model_csv(model, "decisions/latest_decision.csv"),
        "portfolio": _normalize_portfolio_columns(load_model_csv(model, "portfolio/portfolio.csv")),
        "target_weights": load_model_csv(model, "target_weights/latest_target_weights.csv"),
        "tw_history": load_model_csv(model, "target_weights/target_weights.csv"),
        "positions": load_model_csv(model, "positions/latest_positions.csv"),
        "planned_orders": load_model_csv(model, "orders/latest_planned_orders.csv"),
        "submitted_orders": load_model_csv(model, "orders/latest_submitted_orders.csv"),
        "orders_history": load_model_csv(model, "orders/submitted_orders.csv"),
        "signal_history": load_model_csv(model, "health/signal_history.csv"),
        "health_status": load_model_health(model),
    }


# =============================================================================
# HELPERS
# =============================================================================

def fmt_dollars(v):
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "—"


def fmt_pct(v):
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "—"


def fmt_pct_signed(v):
    try:
        return f"{float(v) * 100:+.2f}%"
    except Exception:
        return "—"


def fmt_num(v, digits=2):
    try:
        if pd.isna(v) or not np.isfinite(float(v)):
            return "—"
        return f"{float(v):,.{digits}f}"
    except Exception:
        return "—"


def hex_to_rgba(hex_color, alpha=0.08):
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return f"rgba(76,158,255,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def metric_card(label, value, delta=None, delta_sign=None):
    delta_html = ""
    if delta is not None:
        cls = f"delta-{delta_sign or 'neu'}"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=None):
    sub = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        {sub}
        """,
        unsafe_allow_html=True,
    )


def action_badge(action):
    color = ACTION_COLORS.get(str(action).lower(), PALETTE["muted"])
    st.markdown(
        f"""
        <span class="action-badge" style="color:{color}; border-color:{color}55;">
            {action}
        </span>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status):
    status = str(status or "").lower()
    if status == "connected":
        return '<span class="status-live">LIVE PAPER</span>'
    if status == "dry_run":
        return '<span class="status-dry">DRY RUN</span>'
    if status in {"healthy", "ok"}:
        return '<span class="status-live">HEALTHY</span>'
    if status in {"warning", "pending"}:
        return '<span class="status-dry">WATCH</span>'
    if status in {"degraded", "error", "failed"}:
        return '<span class="status-error">DEGRADED</span>'
    return '<span class="status-pending">PENDING</span>'


def _interpret_health(health_dict):
    if not health_dict:
        return ("Pending first check", "pending", True)

    overall = (health_dict or {}).get("overall_status", "unknown")
    n_decisions = (health_dict or {}).get("n_decisions") or 0

    if n_decisions < 5:
        return ("Pending first check", "pending", True)

    if overall in {"healthy", "warning", "degraded"}:
        return (overall.title(), overall, False)

    return ("Pending first check", "pending", True)


def chart_layout(fig, title=None, height=420):
    fig.update_layout(
        title=dict(text=title or "", x=0.02, xanchor="left", font=dict(size=18, color=PALETTE["text"])),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,11,18,0.2)",
        font=dict(color="#dbe4f0", family="Inter"),
        margin=dict(l=28, r=28, t=54 if title else 28, b=28),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="#94a3b8",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            color="#94a3b8",
        ),
        hovermode="x unified",
    )
    return fig


def compute_perf_stats(daily_returns: pd.Series, min_obs: int = 5) -> dict:
    nan_result = dict(
        total_return=np.nan,
        ann_return=np.nan,
        ann_vol=np.nan,
        sharpe=np.nan,
        sortino=np.nan,
        calmar=np.nan,
        max_dd=np.nan,
        hit_rate=np.nan,
        t_stat=np.nan,
        profit_factor=np.nan,
        avg_win=np.nan,
        avg_loss=np.nan,
        win_loss_ratio=np.nan,
        best_day=np.nan,
        worst_day=np.nan,
        n_days=0,
        n_win_days=0,
        n_loss_days=0,
        longest_dd_days=0,
        current_dd=np.nan,
        current_dd_days=0,
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

    dd_series = eq / eq.cummax() - 1.0
    max_dd = float(dd_series.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else np.nan

    hit_rate = float((x > 0).mean())
    wins = x[x > 0]
    losses = x[x < 0]

    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(losses.mean()) if len(losses) else np.nan
    win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss and not pd.isna(avg_loss) and avg_loss != 0 else np.nan
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan

    in_dd = dd_series < 0
    longest_dd = 0
    cur_run = 0
    for v in in_dd.values:
        if v:
            cur_run += 1
            longest_dd = max(longest_dd, cur_run)
        else:
            cur_run = 0

    current_dd = float(dd_series.iloc[-1])
    current_dd_days = 0
    for v in dd_series.iloc[::-1].values:
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
        best_day=float(x.max()),
        worst_day=float(x.min()),
        n_days=len(x),
        n_win_days=int((x > 0).sum()),
        n_loss_days=int((x < 0).sum()),
        longest_dd_days=longest_dd,
        current_dd=current_dd,
        current_dd_days=current_dd_days,
    )


@st.cache_data(ttl=900)
def fetch_benchmark_returns(ticker: str, start: str, end: str = None) -> pd.Series:
    try:
        import yfinance as yf

        kwargs = dict(start=start, auto_adjust=True, progress=False)
        if end:
            kwargs["end"] = end

        data = yf.download(ticker, **kwargs)
        if data is None or data.empty:
            return pd.Series(dtype=float)

        if isinstance(data.columns, pd.MultiIndex):
            try:
                close = data[("Close", ticker)]
            except KeyError:
                close = data["Close"].iloc[:, 0] if "Close" in data.columns.get_level_values(0) else data.iloc[:, 0]
        else:
            close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = pd.Series(close).dropna().astype(float)
        if close.empty:
            return pd.Series(dtype=float)

        idx = pd.to_datetime(close.index)
        try:
            idx = idx.tz_convert(None)
        except Exception:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass

        close.index = pd.to_datetime(idx).normalize()
        rets = close.pct_change().dropna()
        rets.name = ticker
        return rets
    except Exception:
        return pd.Series(dtype=float)


def _to_date_index(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return s

    out = s.copy()
    idx = pd.to_datetime(out.index)

    try:
        idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass

    out.index = pd.to_datetime(idx).normalize()
    out = out[~out.index.duplicated(keep="last")]
    return out


def get_portfolio_series(portfolio_df):
    if portfolio_df is None or portfolio_df.empty or "portfolio_value" not in portfolio_df.columns:
        return pd.Series(dtype=float)

    df = portfolio_df.copy()
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"])
        if df.empty:
            return pd.Series(dtype=float)
        s = pd.Series(pd.to_numeric(df["portfolio_value"], errors="coerce").values, index=df["timestamp_utc"])
    else:
        s = pd.Series(pd.to_numeric(df["portfolio_value"], errors="coerce").values)

    return s.dropna()


def portfolio_returns(portfolio_df):
    s = get_portfolio_series(portfolio_df)
    if len(s) < 2:
        return pd.Series(dtype=float)
    return s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def latest_value(df, column, default=None):
    if df is None or df.empty or column not in df.columns:
        return default
    try:
        return df[column].iloc[-1]
    except Exception:
        return default


def clean_table(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in out.columns:
        if "timestamp" in c.lower():
            out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)
    return out


# =============================================================================
# LOAD SELECTED MODEL
# =============================================================================

registry = load_models_registry()
model_ids = [m["id"] for m in registry]
model_names = {m["id"]: m.get("name", m["id"]) for m in registry}
model_color = {m["id"]: m.get("color", PALETTE["blue"]) for m in registry}


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">QSentia</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">BR-PPO Terminal</div>', unsafe_allow_html=True)

    selected_id = st.selectbox(
        "Model",
        options=model_ids,
        index=0,
        format_func=lambda mid: model_names.get(mid, mid),
        key="selected_model",
    )

    selected_model = next((m for m in registry if m["id"] == selected_id), registry[0])

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    data = load_all_for_model(selected_model)
    dec = data["latest_decision"]
    dec_hist = data["decisions"]
    port = data["portfolio"]

    src_repo = selected_model.get("repo")
    src_path = selected_model.get("logs_path") or f"logs/{selected_id}"

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Selected Strategy</div>
            <div class="sidebar-value">{model_names.get(selected_id, selected_id)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if src_repo:
        source_text = f"{src_repo}<br>{src_path}"
    else:
        source_text = src_path

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Data Source</div>
            <div class="sidebar-value">{source_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    last_run = None
    if not dec.empty and "timestamp_utc" in dec.columns:
        last_run = pd.to_datetime(dec["timestamp_utc"].iloc[-1], utc=True, errors="coerce")

    if last_run is not None and pd.notna(last_run):
        now_utc = datetime.now(timezone.utc)
        delta_m = int((now_utc - last_run).total_seconds() / 60)
        if delta_m < 60:
            last_run_ago = f"{delta_m} minutes ago"
        elif delta_m < 1440:
            last_run_ago = f"{delta_m // 60}h {delta_m % 60}m ago"
        else:
            last_run_ago = f"{delta_m // 1440} days ago"

        last_run_text = f"{last_run.strftime('%b %d, %H:%M UTC')}<br><span class='small-muted'>{last_run_ago}</span>"
    else:
        last_run_text = "No data yet"

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Last Workflow Run</div>
            <div class="sidebar-value">{last_run_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    account_status = latest_value(dec, "account_status", "unknown")
    submit_orders = str(latest_value(dec, "submit_orders", "false")).lower()
    current_action = latest_value(dec, "action", "—")
    latest_portfolio = latest_value(port, "portfolio_value", None)

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Trading State</div>
            <div class="sidebar-value">{status_badge(account_status)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Order Submission</div>
            <div class="sidebar-value">{"Enabled" if submit_orders == "true" else "Disabled"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Latest Portfolio</div>
            <div class="sidebar-value">{fmt_dollars(latest_portfolio)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Current Allocation Signal</div>
            <div class="sidebar-value">{current_action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Total Decisions</div>
            <div class="sidebar-value">{len(dec_hist) if dec_hist is not None and not dec_hist.empty else 0}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer-note">
        Research and paper trading environment. Not investment advice.
        Dashboard refreshes cached data every two minutes.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HEADER
# =============================================================================

selected_name = model_names.get(selected_id, selected_id)
health_label, health_state, health_pending = _interpret_health(data.get("health_status", {}))

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">QSentia Research Terminal</div>
        <div class="hero-title">BR-PPO Paper Trading Monitor</div>
        <div class="hero-subtitle">
            Investor-facing command center for reinforcement learning based portfolio allocation.
            Monitor live paper trading behavior, model comparison, portfolio exposure, execution logs,
            and health diagnostics from the committed trading workflow.
        </div>
        <div class="badge-row">
            <span class="badge">Selected Model: {selected_name}</span>
            <span class="badge">Environment: Alpaca Paper Trading</span>
            <span class="badge">Data: GitHub Actions Logs</span>
            <span class="badge">Health: {health_label}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# TOP KPI STRIP
# =============================================================================

port_rets = portfolio_returns(port)
perf = compute_perf_stats(port_rets)

latest_portfolio = latest_value(port, "portfolio_value", None)
first_portfolio = None
if port is not None and not port.empty and "portfolio_value" in port.columns:
    first_portfolio = pd.to_numeric(port["portfolio_value"], errors="coerce").dropna()
    first_portfolio = first_portfolio.iloc[0] if len(first_portfolio) else None

pnl = None
pnl_pct = None
if latest_portfolio is not None and first_portfolio not in [None, 0]:
    try:
        pnl = float(latest_portfolio) - float(first_portfolio)
        pnl_pct = float(latest_portfolio) / float(first_portfolio) - 1.0
    except Exception:
        pass

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    metric_card("Portfolio Value", fmt_dollars(latest_portfolio), fmt_pct_signed(pnl_pct) if pnl_pct is not None else None, "pos" if pnl_pct and pnl_pct >= 0 else "neg")
with k2:
    metric_card("Total P&L", fmt_dollars(pnl) if pnl is not None else "—", "Paper account basis", "neu")
with k3:
    metric_card("Annualized Sharpe", fmt_num(perf.get("sharpe"), 2), f"{perf.get('n_days', 0)} return observations", "neu")
with k4:
    metric_card("Max Drawdown", fmt_pct_signed(perf.get("max_dd")), "Peak to trough", "neg")
with k5:
    metric_card("Current Signal", str(current_action), "Latest model allocation", "neu")


# =============================================================================
# TABS
# =============================================================================

tab_compare, tab_overview, tab_perf, tab_portfolio, tab_orders, tab_history, tab_health = st.tabs(
    [
        "Model Comparison",
        "Executive Overview",
        "Performance Analytics",
        "Portfolio Exposure",
        "Execution Monitor",
        "Decision History",
        "Model Health",
    ]
)


# =============================================================================
# TAB 1: MODEL COMPARISON
# =============================================================================

with tab_compare:
    section_header(
        "Model Comparison",
        "Side by side normalized equity curves for every enabled strategy, benchmarked against major U.S. market indices.",
    )

    benchmark_map = {
        "S&P 500": {"ticker": "SPY", "color": "#ffd166"},
        "Nasdaq 100": {"ticker": "QQQ", "color": "#a78bfa"},
        "Dow Jones": {"ticker": "DIA", "color": "#ff4b6e"},
        "Russell 2000": {"ticker": "IWM", "color": "#f97316"},
        "Total U.S. Market": {"ticker": "VTI", "color": "#94a3b8"},
    }

    fig = go.Figure()
    summary_rows = []
    portfolio_start_dates = []
    portfolio_end_dates = []

    for model in registry:
        mid = model["id"]
        mname = model.get("name", mid)
        color = model.get("color", PALETTE["blue"])
        mdata = load_all_for_model(model)
        mport = mdata.get("portfolio", pd.DataFrame())
        s = get_portfolio_series(mport)

        if len(s) >= 2:
            s = s.sort_index()
            portfolio_start_dates.append(pd.to_datetime(s.index.min()).date())
            portfolio_end_dates.append(pd.to_datetime(s.index.max()).date())

            norm = s / s.iloc[0] * 100.0

            fig.add_trace(
                go.Scatter(
                    x=norm.index,
                    y=norm.values,
                    mode="lines",
                    name=mname,
                    line=dict(width=4, color=color),
                    hovertemplate=f"{mname}<br>Date=%{{x}}<br>Value=%{{y:.2f}}<extra></extra>",
                )
            )

            r = s.pct_change().dropna()
            p = compute_perf_stats(r)

            summary_rows.append(
                {
                    "Asset": mname,
                    "Type": "Strategy",
                    "Latest Value": fmt_dollars(s.iloc[-1]),
                    "Total Return": fmt_pct_signed(s.iloc[-1] / s.iloc[0] - 1.0),
                    "Sharpe": fmt_num(p.get("sharpe"), 2),
                    "Max Drawdown": fmt_pct_signed(p.get("max_dd")),
                    "Observations": p.get("n_days", 0),
                }
            )

    if portfolio_start_dates:
        benchmark_start = min(portfolio_start_dates).strftime("%Y-%m-%d")
        benchmark_end = None
        if portfolio_end_dates:
            benchmark_end = max(portfolio_end_dates).strftime("%Y-%m-%d")

        for bench_name, bench_info in benchmark_map.items():
            ticker = bench_info["ticker"]
            color = bench_info["color"]

            bench_rets = fetch_benchmark_returns(
                ticker=ticker,
                start=benchmark_start,
                end=benchmark_end,
            )

            if bench_rets is not None and len(bench_rets) > 0:
                bench_index = (1.0 + bench_rets).cumprod() * 100.0
                bench_index = _to_date_index(bench_index)

                fig.add_trace(
                    go.Scatter(
                        x=bench_index.index,
                        y=bench_index.values,
                        mode="lines",
                        name=f"{bench_name} ({ticker})",
                        line=dict(
                            width=2.4,
                            color=color,
                            dash="dash",
                        ),
                        opacity=0.82,
                        hovertemplate=f"{bench_name} ({ticker})<br>Date=%{{x}}<br>Value=%{{y:.2f}}<extra></extra>",
                    )
                )

                bp = compute_perf_stats(bench_rets)

                summary_rows.append(
                    {
                        "Asset": f"{bench_name} ({ticker})",
                        "Type": "Benchmark",
                        "Latest Value": "Index = 100",
                        "Total Return": fmt_pct_signed(bench_index.iloc[-1] / bench_index.iloc[0] - 1.0),
                        "Sharpe": fmt_num(bp.get("sharpe"), 2),
                        "Max Drawdown": fmt_pct_signed(bp.get("max_dd")),
                        "Observations": bp.get("n_days", 0),
                    }
                )

    if fig.data:
        chart_layout(
            fig,
            "",
            height=560,
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", size=11),
            )
        )
        fig.update_yaxes(title="Normalized Value")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="small-muted">
            Solid lines represent BR-PPO strategy portfolios. Dashed lines represent benchmark ETFs normalized to the same starting value.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("No portfolio history is available for model comparison yet.")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        strategy_rows = summary_df[summary_df["Type"] == "Strategy"]
        benchmark_rows = summary_df[summary_df["Type"] == "Benchmark"]

        section_header(
            "Relative Performance Table",
            "Strategies and benchmarks are normalized for visual comparison. Returns and risk metrics are calculated from available observations.",
        )

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if not strategy_rows.empty and not benchmark_rows.empty:
            st.markdown("#### Benchmark Set Included")
            st.dataframe(
                benchmark_rows[["Asset", "Total Return", "Sharpe", "Max Drawdown", "Observations"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No strategy or benchmark rows are available yet.")

# =============================================================================
# TAB 2: EXECUTIVE OVERVIEW
# =============================================================================

with tab_overview:
    section_header(
        "Executive Overview",
        "A concise view of the current model state, account status, latest allocation signal, and portfolio trajectory.",
    )

    c1, c2 = st.columns([1.35, 0.65])

    with c1:
        s = get_portfolio_series(port)
        if len(s) >= 2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color=PALETTE["green"], width=3),
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(PALETTE["green"], 0.08),
                )
            )
            chart_layout(fig, "Portfolio Value Over Time", height=460)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Portfolio history will appear after at least two logged portfolio observations.")

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Current Operating State</div>', unsafe_allow_html=True)
        st.markdown(f"**Account:** {status_badge(account_status)}", unsafe_allow_html=True)
        st.markdown(f"**Order Submission:** {'Enabled' if submit_orders == 'true' else 'Disabled'}")
        st.markdown("**Current Signal:**")
        action_badge(current_action)
        st.markdown("---")
        st.markdown(f"**Model:** {selected_name}")
        st.markdown(f"**Health:** {status_badge(health_state)}", unsafe_allow_html=True)
        st.markdown(f"**Last Run:** {last_run.strftime('%b %d, %Y %H:%M UTC') if last_run is not None and pd.notna(last_run) else 'No data yet'}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        metric_card("Annualized Return", fmt_pct_signed(perf.get("ann_return")), "Based on logged portfolio series", "pos" if perf.get("ann_return", 0) and perf.get("ann_return", 0) >= 0 else "neg")
    with c4:
        metric_card("Annualized Volatility", fmt_pct(perf.get("ann_vol")), "Risk estimate", "neu")
    with c5:
        metric_card("Hit Rate", fmt_pct(perf.get("hit_rate")), "Positive return days", "neu")
    with c6:
        metric_card("Calmar Ratio", fmt_num(perf.get("calmar"), 2), "Return versus drawdown", "neu")


# =============================================================================
# TAB 3: PERFORMANCE ANALYTICS
# =============================================================================

with tab_perf:
    section_header(
        "Performance Analytics",
        "Risk adjusted performance, drawdown behavior, daily return distribution, and benchmark-ready diagnostics.",
    )

    rets = port_rets
    s = get_portfolio_series(port)

    if len(rets) < 2:
        st.info("Performance analytics will populate after more portfolio observations are logged.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            metric_card("Total Return", fmt_pct_signed(perf.get("total_return")), "Since first observation", "pos" if perf.get("total_return", 0) >= 0 else "neg")
        with p2:
            metric_card("Sortino Ratio", fmt_num(perf.get("sortino"), 2), "Downside risk adjusted", "neu")
        with p3:
            metric_card("Profit Factor", fmt_num(perf.get("profit_factor"), 2), "Gross wins divided by losses", "neu")
        with p4:
            metric_card("Worst Day", fmt_pct_signed(perf.get("worst_day")), "Single period downside", "neg")

        c1, c2 = st.columns(2)

        with c1:
            eq = (1 + rets).cumprod()
            dd = eq / eq.cummax() - 1.0
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd.values,
                    mode="lines",
                    name="Drawdown",
                    line=dict(color=PALETTE["red"], width=2),
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(PALETTE["red"], 0.16),
                )
            )
            chart_layout(fig, "Drawdown Profile", height=420)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=rets.values,
                    nbinsx=30,
                    name="Returns",
                    marker=dict(color=PALETTE["blue"], line=dict(color="rgba(255,255,255,0.12)", width=1)),
                )
            )
            chart_layout(fig, "Return Distribution", height=420)
            fig.update_xaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)

        stats_df = pd.DataFrame(
            [
                ["Total Return", fmt_pct_signed(perf.get("total_return"))],
                ["Annualized Return", fmt_pct_signed(perf.get("ann_return"))],
                ["Annualized Volatility", fmt_pct(perf.get("ann_vol"))],
                ["Sharpe Ratio", fmt_num(perf.get("sharpe"), 3)],
                ["Sortino Ratio", fmt_num(perf.get("sortino"), 3)],
                ["Calmar Ratio", fmt_num(perf.get("calmar"), 3)],
                ["Max Drawdown", fmt_pct_signed(perf.get("max_dd"))],
                ["Hit Rate", fmt_pct(perf.get("hit_rate"))],
                ["T Statistic", fmt_num(perf.get("t_stat"), 3)],
                ["Best Day", fmt_pct_signed(perf.get("best_day"))],
                ["Worst Day", fmt_pct_signed(perf.get("worst_day"))],
                ["Current Drawdown", fmt_pct_signed(perf.get("current_dd"))],
                ["Longest Drawdown Days", str(perf.get("longest_dd_days", 0))],
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: PORTFOLIO EXPOSURE
# =============================================================================

with tab_portfolio:
    section_header(
        "Portfolio Exposure",
        "Current holdings, target weights, and historical target allocation produced by the trading model.",
    )

    target_weights = data.get("target_weights", pd.DataFrame())
    tw_history = data.get("tw_history", pd.DataFrame())
    positions = data.get("positions", pd.DataFrame())

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Current Target Weights</div>', unsafe_allow_html=True)
        if target_weights is not None and not target_weights.empty:
            df = target_weights.copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            symbol_col = None
            for candidate in ["symbol", "ticker", "asset"]:
                if candidate in df.columns:
                    symbol_col = candidate
                    break

            weight_col = None
            for candidate in ["target_weight", "weight", "target"]:
                if candidate in df.columns:
                    weight_col = candidate
                    break

            if symbol_col and weight_col:
                fig = px.pie(
                    df,
                    names=symbol_col,
                    values=weight_col,
                    hole=0.62,
                    title=None,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                chart_layout(fig, "Target Allocation", height=390)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(clean_table(df), use_container_width=True, hide_index=True)
        else:
            st.info("No target weights logged yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Current Positions</div>', unsafe_allow_html=True)
        if positions is not None and not positions.empty:
            st.dataframe(clean_table(positions), use_container_width=True, hide_index=True)
        else:
            st.info("No positions logged yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    if tw_history is not None and not tw_history.empty:
        section_header("Target Weight History", "Historical model allocation changes over time.")
        df = tw_history.copy()
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

            symbol_col = next((c for c in ["symbol", "ticker", "asset"] if c in df.columns), None)
            weight_col = next((c for c in ["target_weight", "weight", "target"] if c in df.columns), None)

            if symbol_col and weight_col:
                fig = px.line(
                    df.dropna(subset=["timestamp_utc"]),
                    x="timestamp_utc",
                    y=weight_col,
                    color=symbol_col,
                    title=None,
                )
                chart_layout(fig, "Historical Target Weights", height=460)
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(clean_table(df), use_container_width=True, hide_index=True)
        else:
            st.dataframe(clean_table(df), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 5: EXECUTION MONITOR
# =============================================================================

with tab_orders:
    section_header(
        "Execution Monitor",
        "Planned and submitted orders from the latest paper trading workflow.",
    )

    planned_orders = data.get("planned_orders", pd.DataFrame())
    submitted_orders = data.get("submitted_orders", pd.DataFrame())
    orders_history = data.get("orders_history", pd.DataFrame())

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Latest Planned Orders</div>', unsafe_allow_html=True)
        if planned_orders is not None and not planned_orders.empty:
            st.dataframe(clean_table(planned_orders), use_container_width=True, hide_index=True)
        else:
            st.info("No planned orders logged yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Latest Submitted Orders</div>', unsafe_allow_html=True)
        if submitted_orders is not None and not submitted_orders.empty:
            st.dataframe(clean_table(submitted_orders), use_container_width=True, hide_index=True)
        else:
            st.info("No submitted orders logged yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    section_header("Order History", "All submitted order records available from the committed logs.")
    if orders_history is not None and not orders_history.empty:
        st.dataframe(clean_table(orders_history), use_container_width=True, hide_index=True)
    else:
        st.info("No historical submitted orders logged yet.")


# =============================================================================
# TAB 6: DECISION HISTORY
# =============================================================================

with tab_history:
    section_header(
        "Decision History",
        "Chronological model decisions, allocation signals, and account state from the trading workflow.",
    )

    if dec_hist is not None and not dec_hist.empty:
        df = clean_table(dec_hist)

        if "timestamp_utc" in df.columns and "action" in df.columns:
            action_counts = df["action"].astype(str).value_counts().reset_index()
            action_counts.columns = ["Action", "Count"]

            c1, c2 = st.columns([0.55, 0.45])

            with c1:
                st.dataframe(df.sort_values("timestamp_utc", ascending=False), use_container_width=True, hide_index=True)

            with c2:
                fig = px.bar(action_counts, x="Action", y="Count", title=None)
                chart_layout(fig, "Allocation Signal Frequency", height=430)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No decision history logged yet.")


# =============================================================================
# TAB 7: MODEL HEALTH
# =============================================================================

with tab_health:
    section_header(
        "Model Health",
        "Operational health, signal quality artifacts, and available health monitor outputs.",
    )

    health = data.get("health_status", {}) or {}
    signal_history = data.get("signal_history", pd.DataFrame())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Overall Status", health_label, "Central health monitor", "neu")
    with c2:
        metric_card("Decision Count", str(health.get("n_decisions", len(dec_hist) if dec_hist is not None else 0)), "Minimum history required", "neu")
    with c3:
        metric_card("Signal History Rows", str(len(signal_history) if signal_history is not None else 0), "Logged health observations", "neu")
    with c4:
        metric_card("Portfolio Observations", str(len(port) if port is not None else 0), "Logged equity records", "neu")

    c5, c6 = st.columns([0.45, 0.55])

    with c5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Health Status JSON</div>', unsafe_allow_html=True)
        if health:
            st.json(health)
        else:
            st.info("No health_status.json available yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Signal History</div>', unsafe_allow_html=True)
        if signal_history is not None and not signal_history.empty:
            df = clean_table(signal_history)
            st.dataframe(df, use_container_width=True, hide_index=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "timestamp_utc" in df.columns and numeric_cols:
                selected_metric = st.selectbox("Signal metric", numeric_cols, key="signal_metric_selector")
                fig = px.line(df, x="timestamp_utc", y=selected_metric, title=None)
                chart_layout(fig, f"Signal Metric: {selected_metric}", height=360)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No signal history logged yet.")
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div class="footer-note">
    QSentia BR-PPO Research Terminal. Paper trading dashboard for research, monitoring, and investor demonstration purposes only.
    This interface does not alter the trading backend, model logic, data loaders, account credentials, or execution workflow.
    </div>
    """,
    unsafe_allow_html=True,
)
