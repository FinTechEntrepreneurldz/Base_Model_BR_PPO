"""
BR-PPO Signal Monitor
Computes model health metrics from trade logs.
Writes logs/health_status.json and exits with code 1 if degraded.

Usage:
    python scripts/signal_monitor.py [--days 63] [--threshold-sharpe -0.3] [--threshold-entropy 0.4]
"""

import sys
import json
import math
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import os
PROJECT_ROOT = Path(__file__).parent.parent

def resolve_log_dir(model_id=None):
    """Determine which logs directory to read/write from.

    Priority: explicit --model-id arg > BRPPO_LOG_DIR env > BRPPO_MODEL_ID env > legacy 'logs/'.
    """
    if model_id:
        return PROJECT_ROOT / "logs" / model_id
    env_log_dir = os.environ.get("BRPPO_LOG_DIR")
    if env_log_dir:
        return Path(env_log_dir).expanduser()
    env_model_id = os.environ.get("BRPPO_MODEL_ID")
    if env_model_id:
        return PROJECT_ROOT / "logs" / env_model_id
    return PROJECT_ROOT / "logs"

# Default LOG_DIR resolved from environment; main() may rebind based on --model-id.
LOG_DIR = resolve_log_dir()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_utc_ts(value):
    """Coerce a value (Series or scalar) to tz-aware UTC datetime/timestamp.

    Handles three failure modes that pd.to_datetime(..., utc=True) chokes on:
      1. Already tz-aware Timestamp/Series  — newer pandas refuses utc=True on these
      2. Underscore-separated formats (Model C writes '2026-05-01_15:08:50')
      3. Mixed format Series                — uses errors='coerce' to drop bad rows

    Always returns tz-aware UTC.
    """
    import pandas as _pd
    if isinstance(value, _pd.Series):
        s = value.astype(str).str.strip().str.replace("_", "T", regex=False)
        out = _pd.to_datetime(s, utc=True, errors="coerce")
        return out
    # scalar path
    if isinstance(value, _pd.Timestamp):
        if value.tz is None:
            return value.tz_localize("UTC")
        return value.tz_convert("UTC")
    s = str(value).strip().replace("_", "T")
    return _pd.to_datetime(s, utc=True, errors="coerce")


def safe_float(v, default=0.0):
    try:
        return float(v) if np.isfinite(float(v)) else default
    except Exception:
        return default


def compound_return(rets):
    rets = pd.Series(rets).replace([np.inf, -np.inf], np.nan).dropna()
    return float((1 + rets).prod() - 1) if len(rets) else 0.0


def annualized_sharpe(rets, periods=252):
    rets = pd.Series(rets).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 5 or rets.std() == 0:
        return None
    return float(rets.mean() / rets.std() * math.sqrt(periods))


def shannon_entropy(counts):
    """Normalised Shannon entropy [0,1] of action distribution."""
    counts = np.array([c for c in counts if c > 0], dtype=float)
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    H = -float(np.sum(p * np.log(p + 1e-12)))
    H_max = math.log(len(counts)) if len(counts) > 1 else 1.0
    return H / H_max


def fetch_benchmark_returns(ticker="SPY", days=90):
    """Download benchmark daily returns via yfinance if available."""
    try:
        import yfinance as yf
        period = f"{max(days + 30, 90)}d"
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        closes = df["Close"] if isinstance(df.columns, pd.Index) and "Close" in df.columns else df
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        rets = closes.pct_change().dropna()
        rets.index = pd.to_datetime(rets.index)
        return rets
    except Exception:
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Load logs
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_safe(path):
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        df = pd.read_csv(p)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = _to_utc_ts(df["timestamp_utc"])
        return df
    except Exception:
        return pd.DataFrame()


def load_logs():
    decisions = load_csv_safe(LOG_DIR / "decisions" / "decisions.csv")
    portfolio  = load_csv_safe(LOG_DIR / "portfolio" / "portfolio.csv")
    return decisions, portfolio


# ─────────────────────────────────────────────────────────────────────────────
# Core health computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_health(decisions, portfolio, lookback_days=63):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    status = {
        "computed_at":     now.isoformat(),
        "lookback_days":   lookback_days,
        "overall_status":  "healthy",          # healthy | warning | degraded
        "alerts":          [],
        "metrics":         {},
        "action_counts":   {},
        "action_entropy":  None,
        "portfolio_sharpe_30d":  None,
        "spy_sharpe_30d":        None,
        "portfolio_return_30d":  None,
        "spy_return_30d":        None,
        "n_decisions":           0,
        "n_unique_actions":      0,
        "top_action":            None,
        "top_action_pct":        None,
        "days_since_last_run":   None,
        "training_recommended":  False,
    }

    # ── Days since last run ──
    if not decisions.empty and "timestamp_utc" in decisions.columns:
        last_ts = _to_utc_ts(decisions["timestamp_utc"].max())
        days_since = (now - last_ts).total_seconds() / 86400
        status["days_since_last_run"] = round(days_since, 1)
        if days_since > 5:
            status["alerts"].append(f"No trading run in {days_since:.0f} days (expected daily Mon–Fri).")

    # ── Filter to lookback window ──
    if not decisions.empty and "timestamp_utc" in decisions.columns:
        recent_dec = decisions[decisions["timestamp_utc"] >= cutoff].copy()
    else:
        recent_dec = pd.DataFrame()

    status["n_decisions"] = len(recent_dec)

    # ── Action diversity ──
    if not recent_dec.empty and "action" in recent_dec.columns:
        actions = recent_dec["action"].dropna()
        counts  = actions.value_counts().to_dict()
        status["action_counts"]  = counts
        status["n_unique_actions"] = len(counts)

        entropy = shannon_entropy(list(counts.values()))
        status["action_entropy"] = round(entropy, 4)

        if actions.shape[0] > 0:
            top = actions.value_counts().index[0]
            top_pct = counts[top] / actions.shape[0]
            status["top_action"]     = top
            status["top_action_pct"] = round(float(top_pct), 4)

            if top_pct > 0.85 and len(counts) == 1:
                status["alerts"].append(
                    f"ACTION LOCK-IN: PPO chose '{top}' in {top_pct*100:.0f}% of last {len(actions)} decisions. "
                    "Model may be stuck — consider retraining."
                )
                status["overall_status"] = "degraded"
                status["training_recommended"] = True
            elif top_pct > 0.70:
                status["alerts"].append(
                    f"Low action diversity: '{top}' used in {top_pct*100:.0f}% of recent decisions. Monitor closely."
                )
                if status["overall_status"] == "healthy":
                    status["overall_status"] = "warning"
    else:
        status["alerts"].append("No decision history found. Cannot evaluate action diversity.")

    # ── Portfolio returns ──
    portfolio_rets_30 = pd.Series(dtype=float)
    if not portfolio.empty and "timestamp_utc" in portfolio.columns and "portfolio_value" in portfolio.columns:
        port = portfolio.copy()
        port["timestamp_utc"] = _to_utc_ts(port["timestamp_utc"])
        port = port.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
        port = port.set_index("timestamp_utc")["portfolio_value"].astype(float)

        # Daily returns (one data point per trading day)
        port_daily = port.resample("D").last().ffill().dropna()
        port_daily_ret = port_daily.pct_change().dropna()

        cutoff_30 = now - timedelta(days=30)
        portfolio_rets_30 = port_daily_ret[port_daily_ret.index >= cutoff_30]

        if len(portfolio_rets_30) >= 10:
            p_ret_30 = compound_return(portfolio_rets_30)
            p_sh_30  = annualized_sharpe(portfolio_rets_30)
            status["portfolio_return_30d"] = round(p_ret_30, 6)
            status["portfolio_sharpe_30d"] = round(p_sh_30, 4) if p_sh_30 is not None else None
        else:
            status["alerts"].append(f"Only {len(portfolio_rets_30)} portfolio data points in last 30 days. Need ≥10 for Sharpe.")

    # ── Benchmark comparison ──
    spy_rets = fetch_benchmark_returns("SPY", days=30)
    if not spy_rets.empty:
        cutoff_30 = pd.Timestamp(now - timedelta(days=30), tz="UTC")
        spy_30 = spy_rets[spy_rets.index >= cutoff_30]
        if len(spy_30) >= 10:
            spy_ret_30 = compound_return(spy_30)
            spy_sh_30  = annualized_sharpe(spy_30)
            status["spy_return_30d"]  = round(spy_ret_30, 6)
            status["spy_sharpe_30d"]  = round(spy_sh_30, 4) if spy_sh_30 is not None else None

    # ── Degradation check: return gap ──
    if status["portfolio_return_30d"] is not None and status["spy_return_30d"] is not None:
        ret_gap = status["portfolio_return_30d"] - status["spy_return_30d"]
        status["metrics"]["return_gap_30d_vs_spy"] = round(ret_gap, 6)
        if ret_gap < -0.10:
            status["alerts"].append(
                f"RETURN LAG: Portfolio underperforming SPY by {abs(ret_gap)*100:.1f}% over 30 days."
            )
            status["training_recommended"] = True
            if status["overall_status"] == "healthy":
                status["overall_status"] = "warning"
        if ret_gap < -0.20:
            status["overall_status"] = "degraded"

    # ── Degradation check: Sharpe gap ──
    if status["portfolio_sharpe_30d"] is not None and status["spy_sharpe_30d"] is not None:
        sh_gap = status["portfolio_sharpe_30d"] - status["spy_sharpe_30d"]
        status["metrics"]["sharpe_gap_30d_vs_spy"] = round(sh_gap, 4)
        if sh_gap < -0.5:
            status["alerts"].append(
                f"SHARPE LAG: Portfolio Sharpe ({status['portfolio_sharpe_30d']:.2f}) "
                f"trailing SPY ({status['spy_sharpe_30d']:.2f}) by {abs(sh_gap):.2f} over 30 days."
            )
            if status["overall_status"] == "healthy":
                status["overall_status"] = "warning"
        if sh_gap < -1.0:
            status["overall_status"] = "degraded"
            status["training_recommended"] = True

    # ── Final training recommendation ──
    if status["overall_status"] == "degraded":
        status["training_recommended"] = True

    return status


# ─────────────────────────────────────────────────────────────────────────────
# Action history enrichment — write richer signal CSV
# ─────────────────────────────────────────────────────────────────────────────

def write_signal_history(decisions, portfolio):
    """Write a merged signals CSV that the Streamlit health tab reads."""
    if decisions.empty:
        return

    df = decisions.copy()
    if "timestamp_utc" not in df.columns:
        return

    df["timestamp_utc"] = _to_utc_ts(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Attach portfolio value changes if available
    if not portfolio.empty and "timestamp_utc" in portfolio.columns and "portfolio_value" in portfolio.columns:
        port = portfolio.copy()
        port["timestamp_utc"] = _to_utc_ts(port["timestamp_utc"])
        port = port.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
        port["daily_return"] = port["portfolio_value"].astype(float).pct_change()
        df = pd.merge_asof(
            df.sort_values("timestamp_utc"),
            port[["timestamp_utc", "portfolio_value", "daily_return"]].sort_values("timestamp_utc"),
            on="timestamp_utc",
            direction="nearest",
            tolerance=pd.Timedelta("2d"),
        )

    out_path = LOG_DIR / "health" / "signal_history.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Signal history written → {out_path} ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BR-PPO Signal Monitor")
    parser.add_argument("--model-id",           type=str,   default=None, help="Model ID to monitor (reads logs/<id>/ )")
    parser.add_argument("--days",               type=int,   default=63,   help="Lookback window (days)")
    parser.add_argument("--threshold-entropy",  type=float, default=0.30, help="Min healthy action entropy")
    parser.add_argument("--threshold-sharpe",   type=float, default=-0.5, help="Min healthy Sharpe gap vs SPY")
    parser.add_argument("--fail-on-degraded",   action="store_true",      help="Exit code 1 if status=degraded")
    args = parser.parse_args()

    # Rebind module-level LOG_DIR if --model-id was passed
    global LOG_DIR
    if args.model_id:
        LOG_DIR = PROJECT_ROOT / "logs" / args.model_id

    print("=" * 60)
    print("BR-PPO Signal Monitor")
    if args.model_id:
        print(f"Model: {args.model_id}")
    print(f"Reading logs from: {LOG_DIR}")
    print("=" * 60)

    decisions, portfolio = load_logs()
    print(f"Loaded {len(decisions)} decision rows, {len(portfolio)} portfolio rows")

    health = compute_health(decisions, portfolio, lookback_days=args.days)
    write_signal_history(decisions, portfolio)

    # Write health status JSON
    out_path = LOG_DIR / "health" / "health_status.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(health, f, indent=2, default=str)

    print(f"\nHealth status: {health['overall_status'].upper()}")
    if health["alerts"]:
        print("\nALERTS:")
        for a in health["alerts"]:
            print(f"  ⚠️  {a}")
    else:
        print("No alerts. Model healthy.")

    if health.get("portfolio_sharpe_30d"):
        print(f"\nPortfolio 30d Sharpe: {health['portfolio_sharpe_30d']:.3f}")
    if health.get("spy_sharpe_30d"):
        print(f"SPY       30d Sharpe: {health['spy_sharpe_30d']:.3f}")
    if health.get("action_entropy") is not None:
        print(f"Action entropy:       {health['action_entropy']:.3f} (1.0 = fully diverse)")

    print(f"\nHealth status written → {out_path}")

    if args.fail_on_degraded and health["overall_status"] == "degraded":
        print("\nExiting with code 1 (degraded).")
        sys.exit(1)

    # Non-zero exit if training recommended (for GitHub Actions to use)
    if health["training_recommended"]:
        sys.exit(2)   # 2 = warning/retraining recommended

    sys.exit(0)


if __name__ == "__main__":
    main()
