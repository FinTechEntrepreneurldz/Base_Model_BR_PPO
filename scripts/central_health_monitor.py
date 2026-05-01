"""
Central Health Monitor (Option 2 architecture)

Fetches logs from EVERY model in models.yaml (via raw GitHub URLs for models
living in separate repos), runs the same compute_health() logic that
signal_monitor.py uses, and writes the resulting health_status.json into
the *dashboard* repo (Base_Model_BR_PPO/logs/<model_id>/health/).

This means:
  * One canonical source of truth for health metrics — the dashboard repo.
  * The dashboard reads health from its own repo (already does for model_a),
    not from each model's own repo.
  * Adding a new model means adding a row to models.yaml — no code changes
    needed in the model's own repo for health tracking.

Usage:
    python scripts/central_health_monitor.py [--lookback-days 63]

Designed to be invoked from a GitHub Actions workflow that runs after the
daily trader matrix.  The workflow then commits the updated health JSONs.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

# Reuse the existing health-computation logic for consistency
THIS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
from signal_monitor import compute_health  # noqa: E402

# Where to write the central health JSONs (inside this repo, the dashboard repo)
DASH_REPO_ROOT = PROJECT_ROOT
LOGS_ROOT      = DASH_REPO_ROOT / "logs"

# Where to find models.yaml (single source of truth)
MODELS_YAML = DASH_REPO_ROOT / "models.yaml"

RAW_BASE = "https://raw.githubusercontent.com"


def _fetch_csv_url(url: str, timeout: int = 30) -> pd.DataFrame:
    """Fetch a CSV from a raw GitHub URL.  Returns empty DataFrame on any failure."""
    headers = {
        "User-Agent": "central-health-monitor/1.0",
        "Cache-Control": "no-cache",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if "timestamp_utc" in df.columns:
            # Some paper traders write underscore-separated timestamps
            # (e.g. "2026-05-01_15:08:50") instead of ISO 8601 "T".  Normalize.
            s = df["timestamp_utc"].astype(str).str.strip().str.replace("_", "T", regex=False)
            df["timestamp_utc"] = pd.to_datetime(s, utc=True, errors="coerce")
        return df
    except Exception as exc:
        print(f"  fetch failed for {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return pd.DataFrame()


def _model_logs_base_url(model: dict) -> str | None:
    """Return the base URL for a model's logs/ folder.

    Returns None for models living locally in the dashboard repo (no `repo` field).
    """
    repo = model.get("repo")
    if not repo:
        return None
    branch    = model.get("branch", "main")
    logs_path = (model.get("logs_path") or f"logs/{model['id']}").strip("/")
    return f"{RAW_BASE}/{repo}/{branch}/{logs_path}"


def _load_local_logs(model: dict):
    """Load decisions + portfolio for a model living in this same repo."""
    logs_path = (model.get("logs_path") or f"logs/{model['id']}").strip("/")
    base = DASH_REPO_ROOT / logs_path
    decisions = pd.DataFrame()
    portfolio = pd.DataFrame()
    dec_path = base / "decisions" / "decisions.csv"
    port_path = base / "portfolio" / "portfolio.csv"
    def _ts(col):
        s = col.astype(str).str.strip().str.replace("_", "T", regex=False)
        return pd.to_datetime(s, utc=True, errors="coerce")
    if dec_path.exists():
        decisions = pd.read_csv(dec_path)
        if "timestamp_utc" in decisions.columns:
            decisions["timestamp_utc"] = _ts(decisions["timestamp_utc"])
    if port_path.exists():
        portfolio = pd.read_csv(port_path)
        if "timestamp_utc" in portfolio.columns:
            portfolio["timestamp_utc"] = _ts(portfolio["timestamp_utc"])
    return decisions, portfolio


def _load_remote_logs(model: dict):
    """Fetch decisions + portfolio for a model living in a separate GitHub repo."""
    base = _model_logs_base_url(model)
    if base is None:
        return _load_local_logs(model)
    decisions = _fetch_csv_url(f"{base}/decisions/decisions.csv")
    portfolio = _fetch_csv_url(f"{base}/portfolio/portfolio.csv")
    return decisions, portfolio


def _normalize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the dashboard's column-aliasing: equity → portfolio_value.

    compute_health() expects portfolio_value; per-repo paper traders may write
    'equity' instead.
    """
    if df is None or df.empty or "portfolio_value" in df.columns:
        return df
    for alt in ("equity", "account_value", "total_equity"):
        if alt in df.columns:
            df = df.copy()
            df["portfolio_value"] = pd.to_numeric(df[alt], errors="coerce")
            return df
    return df


def write_health(model_id: str, health: dict):
    """Write a model's health_status.json into the central dashboard repo."""
    out_dir = LOGS_ROOT / model_id / "health"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "health_status.json"
    with open(path, "w") as f:
        json.dump(health, f, indent=2, default=str)
    print(f"  wrote {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=63)
    ap.add_argument("--only-model",    type=str, default=None,
                    help="If set, compute health for only this model_id.")
    args = ap.parse_args()

    if not MODELS_YAML.exists():
        print(f"ERROR: {MODELS_YAML} not found", file=sys.stderr)
        sys.exit(1)

    with open(MODELS_YAML) as f:
        cfg = yaml.safe_load(f)
    models = cfg.get("models", [])

    if args.only_model:
        models = [m for m in models if m["id"] == args.only_model]
        if not models:
            print(f"ERROR: no model with id={args.only_model}", file=sys.stderr)
            sys.exit(1)

    print(f"Central health monitor — {len(models)} model(s), "
          f"lookback={args.lookback_days} days")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}\n")

    summary = []
    for model in models:
        if not model.get("enabled", True):
            print(f"[SKIP] {model['id']} — disabled in models.yaml")
            continue

        mid = model["id"]
        repo = model.get("repo", "(local)")
        print(f"[{mid}]  source: {repo}")

        try:
            decisions, portfolio = _load_remote_logs(model)
            portfolio = _normalize_portfolio(portfolio)

            n_dec  = len(decisions)
            n_port = len(portfolio)
            print(f"  fetched {n_dec} decisions, {n_port} portfolio rows")

            health = compute_health(decisions, portfolio,
                                     lookback_days=args.lookback_days)
            health["source_repo"] = repo  # provenance
            health["computed_by"] = "central_health_monitor"

            # compute_health() defaults to "healthy" when no failure conditions
            # trigger, but a brand new model with <5 decisions is not actually
            # healthy — it is unknown until enough history accumulates.  Override.
            if (health.get("n_decisions") or 0) < 5:
                health["overall_status"] = "unknown"
                if not health.get("alerts"):
                    health["alerts"] = ["Awaiting more trading runs before health can be assessed."]

            write_health(mid, health)

            summary.append({
                "model_id":       mid,
                "overall_status": health.get("overall_status"),
                "n_decisions":    health.get("n_decisions"),
                "alerts":         len(health.get("alerts") or []),
            })

        except Exception as exc:
            print(f"  ERROR computing health for {mid}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            # Write a failure JSON so the dashboard knows a check happened
            failure = {
                "computed_at":     datetime.now(timezone.utc).isoformat(),
                "lookback_days":   args.lookback_days,
                "overall_status":  "unknown",
                "alerts":          [f"Central monitor failed: {type(exc).__name__}: {exc}"],
                "n_decisions":     0,
                "computed_by":     "central_health_monitor",
                "source_repo":     repo,
            }
            write_health(mid, failure)
            summary.append({
                "model_id":       mid,
                "overall_status": "unknown",
                "n_decisions":    0,
                "alerts":         1,
            })
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in summary:
        print(f"  {s['model_id']:12s}  {s['overall_status']:10s}  "
              f"n_decisions={s['n_decisions']:<4d}  alerts={s['alerts']}")

    # Exit with non-zero only if any model is degraded.  Pending / unknown / hold
    # are not treated as failures — the workflow should stay green for those.
    any_degraded = any(s["overall_status"] == "degraded" for s in summary)
    sys.exit(1 if any_degraded else 0)


if __name__ == "__main__":
    main()
