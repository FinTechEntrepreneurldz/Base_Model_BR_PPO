"""
BR-PPO Strategy Engine
Core trading logic: data download, feature engineering, PPO inference, order management.
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


# ============================================================
# Paths and environment
# ============================================================

PROJECT_ROOT = Path(os.environ.get(
    "BRPPO_PROJECT_ROOT", str(Path(__file__).parent)
)).expanduser()

if load_dotenv is not None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

PROJECT_ROOT = Path(os.environ.get("BRPPO_PROJECT_ROOT", str(PROJECT_ROOT))).expanduser()
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
# LOG_DIR is overridable so multi-model deployments can write to logs/<model_id>/
LOG_DIR      = Path(os.environ.get("BRPPO_LOG_DIR", str(PROJECT_ROOT / "logs"))).expanduser()
DATA_DIR     = PROJECT_ROOT / "data"

# MODEL_ID is informational -- workflows pass it for log labeling
MODEL_ID     = os.environ.get("BRPPO_MODEL_ID", "default")

for folder in [
    ARTIFACT_DIR,
    LOG_DIR,
    LOG_DIR / "decisions",
    LOG_DIR / "orders",
    LOG_DIR / "positions",
    LOG_DIR / "portfolio",
    LOG_DIR / "target_weights",
    DATA_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


def env_bool(name, default=False):
    val = str(os.environ.get(name, str(default))).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def env_int(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


MODEL_PATH    = Path(os.environ.get("BRPPO_MODEL_PATH",    str(ARTIFACT_DIR / "v10_bro_ppo_allocation_agent.zip"))).expanduser()
METADATA_PATH = Path(os.environ.get("BRPPO_METADATA_PATH", str(ARTIFACT_DIR / "v10_bro_ppo_allocation_agent_metadata.json"))).expanduser()

SUBMIT_ORDERS             = env_bool("BRPPO_SUBMIT_ORDERS", True)
MIN_TRADE_DOLLARS         = env_float("BRPPO_MIN_TRADE_DOLLARS", 25)
MAX_POSITION_WEIGHT       = env_float("BRPPO_MAX_POSITION_WEIGHT", 0.20)
MAX_GROSS_EXPOSURE        = env_float("BRPPO_MAX_GROSS_EXPOSURE", 1.00)
CASH_BUFFER_PCT           = env_float("BRPPO_CASH_BUFFER_PCT", 0.02)
DEFAULT_ACCOUNT_VALUE     = env_float("BRPPO_DEFAULT_ACCOUNT_VALUE", 100000)
REBALANCE_THRESHOLD_WEIGHT = env_float("BRPPO_REBALANCE_THRESHOLD_WEIGHT", 0.001)
DATA_PERIOD               = os.environ.get("BRPPO_DATA_PERIOD", "3y")
V6_MAX_NAMES              = env_int("BRPPO_V6_MAX_NAMES", 30)


# ============================================================
# Defaults
# ============================================================

ETF_TICKERS = ["SPY", "QQQ", "VTI", "RSP"]

BLOCKED_TICKERS = {
    t.strip().upper()
    for t in os.environ.get("BRPPO_BLOCKED_TICKERS", "BIL").split(",")
    if t.strip()
}

FORCE_ACTION_NAME = os.environ.get("BRPPO_FORCE_ACTION_NAME", "").strip()

AGGRESSIVE_FALLBACK_ACTIONS = [
    a.strip()
    for a in os.environ.get(
        "BRPPO_AGGRESSIVE_FALLBACK_ACTIONS",
        "v6_alpha,v8_blend,current30_v6_70,current50_v8_30_qqq20,current50_v6_30_qqq20,current70_qqq30,qqq,top_ew,current_ew"
    ).split(",")
    if a.strip()
]

DEFAULT_UNIVERSE = sorted(set([
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "COST",
    "NFLX", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "INTC", "QCOM", "AMAT", "TXN",
    "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "C", "USB",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "ISRG",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX",
    "HD", "LOW", "MCD", "NKE", "SBUX", "TJX", "BKNG", "CMG",
    "PG", "KO", "PEP", "WMT", "CL", "MDLZ",
    "CAT", "GE", "HON", "RTX", "LMT", "UPS", "UNP", "DE",
    "LIN", "SHW", "APD", "ECL",
    "NEE", "DUK", "SO",
]))

DEFAULT_ACTION_SPECS = {
    "current_ew":            {"CURRENT_EW": 1.00},
    "top_ew":                {"TOP_EW": 1.00},
    "spy":                   {"SPY": 1.00},
    "v6_alpha":              {"V6_ALPHA": 1.00},
    "v8_blend":              {"V8_BLEND": 1.00},
    "current70_v6_30":       {"CURRENT_EW": 0.70, "V6_ALPHA": 0.30},
    "current50_v6_50":       {"CURRENT_EW": 0.50, "V6_ALPHA": 0.50},
    "current30_v6_70":       {"CURRENT_EW": 0.30, "V6_ALPHA": 0.70},
    "current60_v6_20_qqq20": {"CURRENT_EW": 0.60, "V6_ALPHA": 0.20, "QQQ": 0.20},
    "current50_v6_30_qqq20": {"CURRENT_EW": 0.50, "V6_ALPHA": 0.30, "QQQ": 0.20},
    "current50_v8_30_qqq20": {"CURRENT_EW": 0.50, "V8_BLEND": 0.30, "QQQ": 0.20},
    "current70_qqq30":       {"CURRENT_EW": 0.70, "QQQ": 0.30},
    "current50_top30_v6_20": {"CURRENT_EW": 0.50, "TOP_EW": 0.30, "V6_ALPHA": 0.20}
}

DEFAULT_ACTION_NAMES = list(DEFAULT_ACTION_SPECS.keys())

DEFAULT_FEATURE_COLS = []
for stream in ["V6_ALPHA", "V8_BLEND", "CURRENT_EW", "TOP_EW", "SPY", "QQQ", "VTI", "RSP"]:
    for w in [21, 63, 126]:
        DEFAULT_FEATURE_COLS.extend([
            f"{stream}_ret_{w}", f"{stream}_vol_{w}",
            f"{stream}_sharpe_{w}", f"{stream}_dd_{w}",
        ])

for spread in ["QQQ_minus_SPY", "V6_minus_CURRENT", "V8_minus_CURRENT"]:
    for w in [21, 63, 126]:
        DEFAULT_FEATURE_COLS.extend([
            f"{spread}_ret_{w}", f"{spread}_vol_{w}",
            f"{spread}_sharpe_{w}", f"{spread}_dd_{w}",
        ])

DEFAULT_FEATURE_COLS.extend(["score_std", "score_spread_90_10", "score_top_decile"])


# ============================================================
# Metadata and model
# ============================================================

def load_metadata():
    meta = {}

    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    action_specs = dict(meta.get("action_specs") or DEFAULT_ACTION_SPECS)

    # Remove any action that allocates to blocked tickers such as BIL.
    if BLOCKED_TICKERS:
        action_specs = {
            action_name: spec
            for action_name, spec in action_specs.items()
            if not any(str(stream).upper() in BLOCKED_TICKERS for stream in spec.keys())
        }

    action_names = [
        action_name
        for action_name in list(meta.get("action_names") or DEFAULT_ACTION_NAMES)
        if action_name in action_specs
    ]

    if not action_names:
        action_specs = {"current_ew": {"CURRENT_EW": 1.0}}
        action_names = ["current_ew"]

    return {
        "raw": meta,
        "feature_cols": list(meta.get("feature_cols") or DEFAULT_FEATURE_COLS),
        "action_names": action_names,
        "action_specs": action_specs,
    }

def choose_fallback_action(metadata):
    action_specs = metadata.get("action_specs", {})
    action_names = metadata.get("action_names", [])

    for action_name in AGGRESSIVE_FALLBACK_ACTIONS:
        if action_name in action_specs and action_name in action_names:
            return action_name, action_names.index(action_name)

    if "v6_alpha" in action_specs and "v6_alpha" in action_names:
        return "v6_alpha", action_names.index("v6_alpha")

    if "v8_blend" in action_specs and "v8_blend" in action_names:
        return "v8_blend", action_names.index("v8_blend")

    if "qqq" in action_specs and "qqq" in action_names:
        return "qqq", action_names.index("qqq")

    if "current_ew" in action_specs and "current_ew" in action_names:
        return "current_ew", action_names.index("current_ew")

    return action_names[0], 0

def load_model():
    if PPO is None:
        raise RuntimeError("stable-baselines3 is not installed.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"PPO model not found: {MODEL_PATH}")
    return PPO.load(str(MODEL_PATH))


def align_observation_to_model(obs, model):
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    try:
        expected = int(model.observation_space.shape[0])
    except Exception:
        return obs

    if len(obs) == expected:
        return obs
    if len(obs) < expected:
        out = np.zeros(expected, dtype=np.float32)
        out[:len(obs)] = obs
        print(f"WARNING: observation padded from {len(obs)} to {expected}.")
        return out
    print(f"WARNING: observation truncated from {len(obs)} to {expected}.")
    return obs[:expected].astype(np.float32)


# ============================================================
# Data
# ============================================================

def download_prices(tickers, period=DATA_PERIOD):
    if yf is None:
        raise RuntimeError("yfinance is not installed.")

    tickers = sorted(set(tickers))
    data = yf.download(
        tickers, period=period, auto_adjust=True,
        group_by="ticker", threads=True, progress=False,
    )

    close  = pd.DataFrame()
    volume = pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close")  in data.columns: close[t]  = data[(t, "Close")]
            if (t, "Volume") in data.columns: volume[t] = data[(t, "Volume")]
    else:
        if len(tickers) == 1:
            if "Close"  in data.columns: close[tickers[0]]  = data["Close"]
            if "Volume" in data.columns: volume[tickers[0]] = data["Volume"]

    close.index  = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    close  = close.sort_index().ffill()
    volume = volume.sort_index().fillna(0)

    return close, volume


# ============================================================
# Feature engineering
# ============================================================

def compound_return(x):
    x = pd.Series(x).dropna()
    return 0.0 if len(x) == 0 else float((1.0 + x).prod() - 1.0)


def max_drawdown(x):
    x = pd.Series(x).dropna()
    if len(x) == 0: return 0.0
    eq = (1.0 + x).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def trailing_features(ret, prefix):
    out = {}
    ret = pd.Series(ret).replace([np.inf, -np.inf], np.nan).dropna()

    for w in [21, 63, 126]:
        x = ret.tail(w)
        if len(x) < 5:
            out[f"{prefix}_ret_{w}"]    = 0.0
            out[f"{prefix}_vol_{w}"]    = 0.0
            out[f"{prefix}_sharpe_{w}"] = 0.0
            out[f"{prefix}_dd_{w}"]     = 0.0
            continue

        vol    = float(x.std() * np.sqrt(252))
        sharpe = float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0

        out[f"{prefix}_ret_{w}"]    = compound_return(x)
        out[f"{prefix}_vol_{w}"]    = vol    if np.isfinite(vol)    else 0.0
        out[f"{prefix}_sharpe_{w}"] = sharpe if np.isfinite(sharpe) else 0.0
        out[f"{prefix}_dd_{w}"]     = max_drawdown(x)

    return out


def ichimoku_score(close):
    close = pd.Series(close).dropna()
    if len(close) < 60: return 0.0

    high   = close; low = close
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)

    score = 0.0
    if close.iloc[-1] > cloud_top.iloc[-1]: score += 1.0
    if tenkan.iloc[-1] > kijun.iloc[-1]:    score += 1.0
    if close.iloc[-1] > close.iloc[-21]:    score += 1.0

    return float(score / 3.0)


def build_v6_alpha_basket(close, volume, max_names=V6_MAX_NAMES):
    if "SPY" not in close.columns:
        return pd.Series(dtype=float)

    symbols = [c for c in close.columns if c not in ETF_TICKERS and close[c].notna().sum() >= 126]
    spy     = close["SPY"].ffill()
    rows    = []

    for sym in symbols:
        px = close[sym].ffill()
        if px.dropna().shape[0] < 126:
            continue
        try:
            r21   = px.pct_change(21).iloc[-1]
            r63   = px.pct_change(63).iloc[-1]
            rs63  = (px / spy).pct_change(63).iloc[-1]
            vol63 = px.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252)
            dd126 = (px / px.rolling(126).max() - 1).iloc[-1]
            ichi  = ichimoku_score(px)
            adv   = (px * volume[sym]).rolling(60).mean().iloc[-1] if sym in volume.columns else 0.0
            liquidity = np.log1p(max(0.0, float(adv))) if np.isfinite(adv) else 0.0

            score = (
                0.30 * np.nan_to_num(r21)
                + 0.35 * np.nan_to_num(r63)
                + 0.45 * np.nan_to_num(rs63)
                + 0.20 * np.nan_to_num(ichi)
                - 0.20 * np.nan_to_num(vol63)
                + 0.10 * np.nan_to_num(dd126)
                + 0.01 * liquidity
            )
            rows.append({"symbol": sym, "score": float(score)})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.Series(dtype=float)

    df = df.sort_values("score", ascending=False).head(max_names)
    return pd.Series(1.0 / len(df), index=df["symbol"].values)


def build_stream_returns(close, volume):
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    streams = pd.DataFrame(index=returns.index)

    for etf in ETF_TICKERS:
        if etf in returns.columns:
            streams[etf] = returns[etf]

    if "RSP" in streams.columns:
        streams["CURRENT_EW"] = streams["RSP"]
        streams["TOP_EW"]     = streams["RSP"]
    elif "SPY" in streams.columns:
        streams["CURRENT_EW"] = streams["SPY"]
        streams["TOP_EW"]     = streams["SPY"]
    else:
        streams["CURRENT_EW"] = 0.0
        streams["TOP_EW"]     = 0.0

    stock_cols = [c for c in returns.columns if c not in ETF_TICKERS]
    streams["V6_ALPHA"] = returns[stock_cols].mean(axis=1) if stock_cols else streams["CURRENT_EW"]

    if "QQQ" in streams.columns:
        streams["V8_BLEND"] = 0.20 * streams["V6_ALPHA"] + 0.50 * streams["CURRENT_EW"] + 0.30 * streams["QQQ"]
    else:
        streams["V8_BLEND"] = streams["V6_ALPHA"]

    for col in ["SPY", "QQQ", "VTI", "RSP"]:
        if col not in streams.columns:
            streams[col] = 0.0

    return streams.fillna(0.0)


def build_live_observation(streams, metadata, last_action_name=None):
    features = {}
    for col in streams.columns:
        features.update(trailing_features(streams[col], col))

    if "QQQ" in streams.columns and "SPY" in streams.columns:
        features.update(trailing_features(streams["QQQ"] - streams["SPY"], "QQQ_minus_SPY"))
    if "V6_ALPHA" in streams.columns and "CURRENT_EW" in streams.columns:
        features.update(trailing_features(streams["V6_ALPHA"] - streams["CURRENT_EW"], "V6_minus_CURRENT"))
    if "V8_BLEND" in streams.columns and "CURRENT_EW" in streams.columns:
        features.update(trailing_features(streams["V8_BLEND"] - streams["CURRENT_EW"], "V8_minus_CURRENT"))

    features.setdefault("score_std", 0.0)
    features.setdefault("score_spread_90_10", 0.0)
    features.setdefault("score_top_decile", 0.0)

    feature_cols = metadata["feature_cols"]
    action_names = metadata["action_names"]

    x = np.array([features.get(c, 0.0) for c in feature_cols], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -10.0, 10.0)

    onehot = np.zeros(len(action_names), dtype=np.float32)
    if last_action_name in action_names:
        onehot[action_names.index(last_action_name)] = 1.0
    else:
        default_action = os.environ.get("BRPPO_DEFAULT_ACTION_NAME", "current_ew")
        if default_action in action_names:
            onehot[action_names.index(default_action)] = 1.0

    obs = np.concatenate([x, onehot]).astype(np.float32)
    return obs, features


def predict_action(model, obs, metadata):
    obs        = align_observation_to_model(obs, model)
    action_idx, _ = model.predict(obs, deterministic=True)
    action_idx = int(action_idx)
    action_names = metadata["action_names"]
    if action_idx < 0 or action_idx >= len(action_names):
        action_idx = action_names.index("current_ew") if "current_ew" in action_names else 0
    return action_names[action_idx], action_idx


def expand_action_to_target_weights(action_name, metadata, close, volume):
    action_specs = metadata["action_specs"]
    if action_name not in action_specs:
        action_name = "current_ew" if "current_ew" in action_specs else list(action_specs.keys())[0]

    spec   = action_specs[action_name]
    target = pd.Series(dtype=float)

    for sleeve, weight in spec.items():
        sleeve = str(sleeve).upper()
        weight = float(weight)

        if sleeve in {"CURRENT_EW", "TOP_EW"}:
            target.loc["RSP"] = target.get("RSP", 0.0) + weight

        elif sleeve == "V6_ALPHA":
            basket = build_v6_alpha_basket(close, volume)
            if basket.empty:
                target.loc["RSP"] = target.get("RSP", 0.0) + weight
            else:
                for sym, w in basket.items():
                    target.loc[sym] = target.get(sym, 0.0) + weight * float(w)

        elif sleeve == "V8_BLEND":
            basket = build_v6_alpha_basket(close, volume)
            if basket.empty:
                target.loc["RSP"] = target.get("RSP", 0.0) + weight * 0.70
                target.loc["QQQ"] = target.get("QQQ", 0.0) + weight * 0.30
            else:
                for sym, w in basket.items():
                    target.loc[sym] = target.get(sym, 0.0) + weight * 0.20 * float(w)
                target.loc["RSP"] = target.get("RSP", 0.0) + weight * 0.50
                target.loc["QQQ"] = target.get("QQQ", 0.0) + weight * 0.30

        elif sleeve in {"SPY", "QQQ", "VTI", "RSP"}:
            target.loc[sleeve] = target.get(sleeve, 0.0) + weight

    if target.empty:
        target.loc["RSP"] = 1.0

    if BLOCKED_TICKERS:
        target = target[~target.index.astype(str).str.upper().isin(BLOCKED_TICKERS)]

    if target.empty:
        target.loc["RSP"] = 1.0

    target = target.groupby(level=0).sum()
    target = target.clip(upper=MAX_POSITION_WEIGHT)

    if target.sum() > 0:
        target = target / target.sum()
        target = target * min(MAX_GROSS_EXPOSURE, 1.0 - CASH_BUFFER_PCT)

    return target.sort_values(ascending=False)


# ============================================================
# Alpaca integration
# ============================================================

def get_alpaca_client():
    try:
        from alpaca.trading.client import TradingClient
    except Exception as exc:
        raise RuntimeError("alpaca-py is not installed.") from exc

    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    paper  = os.environ.get("ALPACA_PAPER", "True").lower() in {"true", "1", "yes"}

    if not key or not secret or key == "YOUR_ALPACA_PAPER_API_KEY":
        raise RuntimeError("Missing Alpaca paper API keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

    return TradingClient(key, secret, paper=paper)


def get_account_value(client=None):
    if client is None:
        try:
            client = get_alpaca_client()
        except Exception:
            return DEFAULT_ACCOUNT_VALUE
    account = client.get_account()
    return float(account.portfolio_value)


def get_positions(client=None):
    if client is None:
        try:
            client = get_alpaca_client()
        except Exception:
            return pd.DataFrame(columns=["symbol", "qty", "market_value", "weight"])

    account_value = get_account_value(client)
    rows = []
    for p in client.get_all_positions():
        mv = float(p.market_value)
        rows.append({
            "symbol":       p.symbol,
            "qty":          float(p.qty),
            "market_value": mv,
            "weight":       mv / account_value if account_value > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def build_order_plan(target_weights, positions, account_value):
    current = (
        positions.set_index("symbol")["weight"].astype(float)
        if positions is not None and not positions.empty
        else pd.Series(dtype=float)
    )

    all_symbols = sorted(set(target_weights.index).union(set(current.index)))
    target  = target_weights.reindex(all_symbols).fillna(0.0)
    current = current.reindex(all_symbols).fillna(0.0)
    delta   = target - current
    rows    = []

    for sym, delta_weight in delta.items():
        if abs(delta_weight) < REBALANCE_THRESHOLD_WEIGHT: continue
        dollars = float(delta_weight * account_value)
        if abs(dollars) < MIN_TRADE_DOLLARS: continue

        rows.append({
            "symbol":         sym,
            "current_weight": float(current.loc[sym]),
            "target_weight":  float(target.loc[sym]),
            "delta_weight":   float(delta_weight),
            "notional":       abs(dollars),
            "side":           "buy" if dollars > 0 else "sell",
        })

    if not rows:
        return pd.DataFrame(columns=["symbol", "current_weight", "target_weight", "delta_weight", "notional", "side"])

    return pd.DataFrame(rows).sort_values("notional", ascending=False).reset_index(drop=True)


def submit_orders(order_plan, client=None):
    if order_plan is None or order_plan.empty:
        return pd.DataFrame()
    if client is None:
        client = get_alpaca_client()

    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    rows = []
    for _, row in order_plan.iterrows():
        side    = OrderSide.BUY if row["side"] == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=row["symbol"],
            notional=round(float(row["notional"]), 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            order = client.submit_order(request)
            rows.append({**row.to_dict(), "submitted": True,  "order_id": str(order.id), "status": str(order.status)})
        except Exception as exc:
            rows.append({**row.to_dict(), "submitted": False, "order_id": None, "status": f"ERROR: {repr(exc)}"})

    return pd.DataFrame(rows)


# ============================================================
# Logging
# ============================================================

def append_csv(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty: return
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log_outputs(decision, target_weights, positions, order_plan, submitted_orders, account_value):
    timestamp = utc_now()

    decision_df = pd.DataFrame([{**decision, "timestamp_utc": timestamp}])

    targets_df = target_weights.reset_index()
    targets_df.columns = ["symbol", "target_weight"]
    targets_df["timestamp_utc"] = timestamp

    portfolio_df = pd.DataFrame([{
        "timestamp_utc":   timestamp,
        "portfolio_value": account_value,
        "action":          decision.get("action"),
        "submit_orders":   SUBMIT_ORDERS,
    }])

    append_csv(LOG_DIR / "decisions"     / "decisions.csv",     decision_df)
    append_csv(LOG_DIR / "target_weights" / "target_weights.csv", targets_df)
    append_csv(LOG_DIR / "portfolio"     / "portfolio.csv",     portfolio_df)

    if positions is not None and not positions.empty:
        pos = positions.copy()
        pos["timestamp_utc"] = timestamp
        append_csv(LOG_DIR / "positions" / "positions.csv", pos)

    if order_plan is not None and not order_plan.empty:
        op = order_plan.copy()
        op["timestamp_utc"] = timestamp
        append_csv(LOG_DIR / "orders" / "planned_orders.csv", op)

    if submitted_orders is not None and not submitted_orders.empty:
        so = submitted_orders.copy()
        so["timestamp_utc"] = timestamp
        append_csv(LOG_DIR / "orders" / "submitted_orders.csv", so)

    # Always write latest snapshots
    decision_df.to_csv(LOG_DIR / "decisions"      / "latest_decision.csv",        index=False)
    targets_df.to_csv( LOG_DIR / "target_weights"  / "latest_target_weights.csv",  index=False)

    if positions is not None:
        positions.to_csv(LOG_DIR / "positions" / "latest_positions.csv", index=False)
    if order_plan is not None:
        order_plan.to_csv(LOG_DIR / "orders" / "latest_planned_orders.csv", index=False)
    if submitted_orders is not None:
        submitted_orders.to_csv(LOG_DIR / "orders" / "latest_submitted_orders.csv", index=False)


# ============================================================
# Main trading cycle
# ============================================================

def run_trading_cycle():
    metadata = load_metadata()

    tickers = sorted(set(DEFAULT_UNIVERSE + ETF_TICKERS))
    print(f"Downloading prices for {len(tickers)} tickers...")
    close, volume = download_prices(tickers, period=DATA_PERIOD)
    print(f"Price data: {close.shape[0]} rows x {close.shape[1]} cols")

    streams = build_stream_returns(close, volume)

    last_action = None
    latest_decision_path = LOG_DIR / "decisions" / "latest_decision.csv"
    if latest_decision_path.exists():
        try:
            last_action = pd.read_csv(latest_decision_path)["action"].iloc[-1]
        except Exception:
            pass

    print(f"Last action: {last_action}")
    obs, feature_values = build_live_observation(streams, metadata, last_action_name=last_action)

    print("Loading PPO model...")
    model = load_model()
    action_name, action_idx = predict_action(model, obs, metadata)

    raw_ppo_action_name = action_name
    raw_ppo_action_idx = action_idx
    reroute_reason = ""
    
    if action_name not in metadata["action_specs"]:
        fallback_action, fallback_idx = choose_fallback_action(metadata)
        reroute_reason = f"blocked_or_invalid_action:{action_name}->aggressive_fallback:{fallback_action}"
        print(f"PPO selected blocked/invalid action {action_name}. Rerouting to aggressive fallback: {fallback_action}")
        action_name = fallback_action
        action_idx = fallback_idx

    if FORCE_ACTION_NAME:
        if FORCE_ACTION_NAME not in metadata["action_specs"]:
            raise ValueError(
                f"BRPPO_FORCE_ACTION_NAME={FORCE_ACTION_NAME} is not available after blocked ticker filtering. "
                f"Available actions: {metadata['action_names'][:20]}"
            )
        print(f"FORCE ACTION OVERRIDE active: {FORCE_ACTION_NAME}")
        action_name = FORCE_ACTION_NAME
        action_idx = metadata["action_names"].index(action_name)

    print(f"PPO action: {action_name} (idx={action_idx})")

    target_weights = expand_action_to_target_weights(action_name, metadata, close, volume)
    print(f"Target positions: {len(target_weights)}")

    try:
        client        = get_alpaca_client()
        account_value = get_account_value(client)
        positions     = get_positions(client)
        account_status = "connected"
        print(f"Account value: ${account_value:,.2f}")
    except Exception as exc:
        print(f"Alpaca unavailable: {exc}. Using dry-run fallback.")
        client         = None
        account_value  = DEFAULT_ACCOUNT_VALUE
        positions      = pd.DataFrame(columns=["symbol", "qty", "market_value", "weight"])
        account_status = "dry_run"

    order_plan = build_order_plan(target_weights, positions, account_value)
    print(f"Planned orders: {len(order_plan)}")

    if SUBMIT_ORDERS and client is not None:
        submitted_orders = submit_orders(order_plan, client=client)
        print(f"Submitted orders: {len(submitted_orders)}")
    else:
        if not SUBMIT_ORDERS:
            print("DRY RUN: BRPPO_SUBMIT_ORDERS=False. No orders submitted.")
        submitted_orders = pd.DataFrame(
            columns=list(order_plan.columns) + ["submitted", "order_id", "status"]
        )

    n_submitted = 0
    if "submitted" in submitted_orders.columns:
        n_submitted = int(submitted_orders["submitted"].sum())

    decision = {
        "market_date":        str(close.index.max().date()) if len(close.index) else None,
        "variant":            "V10_BRO_PPO_AllocationAgent",
        "action":             action_name,
        "action_idx":         action_idx,
        "raw_ppo_action":     raw_ppo_action_name,
        "raw_ppo_action_idx": raw_ppo_action_idx,
        "reroute_reason":     reroute_reason,
        "last_action":        last_action,
        "submit_orders":      SUBMIT_ORDERS,
        "account_status":     account_status,
        "account_value":      account_value,
        "n_target_positions": int(len(target_weights)),
        "n_orders_planned":   int(len(order_plan)),
        "n_orders_submitted": n_submitted,
        "model_path":         str(MODEL_PATH),
        "metadata_path":      str(METADATA_PATH),
    }

    log_outputs(decision, target_weights, positions, order_plan, submitted_orders, account_value)

    return {
        "decision":        decision,
        "target_weights":  target_weights,
        "positions":       positions,
        "order_plan":      order_plan,
        "submitted_orders": submitted_orders,
        "features":        feature_values,
        "account_value":   account_value,
    }


if __name__ == "__main__":
    result = run_trading_cycle()
    print(json.dumps(result["decision"], indent=2))
