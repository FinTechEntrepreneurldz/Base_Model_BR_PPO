"""
BR-PPO Alpaca Paper Trader
Entry point for daily execution. Calls strategy_engine.run_trading_cycle()
and prints a formatted summary. Handles fallback ETF targets gracefully.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import strategy_engine as se
from strategy_engine import run_trading_cycle


# ============================================================
# Helpers
# ============================================================

def _normalize_target_weights(obj):
    """Convert target_weights into a clean DataFrame: columns = symbol, target_weight."""
    if obj is None:
        return pd.DataFrame(columns=["symbol", "target_weight"])

    if isinstance(obj, pd.Series):
        if obj.empty:
            return pd.DataFrame(columns=["symbol", "target_weight"])
        df = obj.reset_index()
        df.columns = ["symbol", "target_weight"]
        return df

    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return pd.DataFrame(columns=["symbol", "target_weight"])
        df = obj.copy()
        if "symbol" in df.columns and "target_weight" in df.columns:
            return df[["symbol", "target_weight"]].copy()
        if "target_weight" in df.columns and df.index.name is not None:
            out = df.reset_index().rename(columns={df.index.name: "symbol"})
            if "symbol" in out.columns:
                return out[["symbol", "target_weight"]].copy()
        if len(df.columns) == 1:
            out = df.reset_index()
            out.columns = ["symbol", "target_weight"]
            return out

    return pd.DataFrame(columns=["symbol", "target_weight"])


def _fallback_target_from_action(action_name):
    """Build a safe ETF proxy if the live basket is empty."""
    action_name = str(action_name or "").lower()

    try:
        metadata  = se.load_metadata()
        specs     = metadata.get("action_specs", {})
        raw_spec  = specs.get(action_name) or specs.get(str(action_name)) or {}
    except Exception:
        raw_spec = {}

    target = {}

    def add(symbol, weight):
        symbol = str(symbol).upper()
        target[symbol] = target.get(symbol, 0.0) + float(weight)

    if raw_spec:
        for sleeve, weight in raw_spec.items():
            sleeve_u = str(sleeve).upper()
            weight   = float(weight)
            if sleeve_u in {"CURRENT_EW", "TOP_EW"}:
                add("RSP", weight)
            elif sleeve_u == "V6_ALPHA":
                add("RSP", weight)
            elif sleeve_u == "V8_BLEND":
                add("RSP", weight * 0.70)
                add("QQQ", weight * 0.30)
            elif sleeve_u in {"SPY", "QQQ", "VTI", "RSP", "BIL"}:
                add(sleeve_u, weight)
    else:
        if "bil" in action_name or "cash" in action_name:
            add("BIL", 1.0)
        elif "qqq" in action_name:
            add("RSP", 0.70); add("QQQ", 0.30)
        elif "spy" in action_name:
            add("SPY", 1.0)
        else:
            add("RSP", 1.0)

    s = pd.Series(target, dtype=float)
    s = s[s > 0]
    if s.empty:
        s = pd.Series({"RSP": 1.0}, dtype=float)

    cash_buffer   = getattr(se, "CASH_BUFFER_PCT", 0.02)
    gross         = min(getattr(se, "MAX_GROSS_EXPOSURE", 1.0), 1.0 - cash_buffer)
    s             = s / s.sum() * gross

    return s.sort_values(ascending=False)


def _rebuild_order_plan(result, target_df):
    decision      = result.get("decision", {})
    account_value = float(decision.get("account_value", getattr(se, "DEFAULT_ACCOUNT_VALUE", 100_000.0)))
    positions     = result.get("positions") or pd.DataFrame(columns=["symbol", "qty", "market_value", "weight"])
    target_series = target_df.set_index("symbol")["target_weight"]

    try:
        return se.build_order_plan(target_series, positions, account_value)
    except Exception as exc:
        print("Could not rebuild order plan:", repr(exc))
        return pd.DataFrame()


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("BR-PPO ALPACA PAPER TRADER — STARTING")
    print("=" * 70)
    model_id = os.environ.get("BRPPO_MODEL_ID", "default")
    print(f"  Model ID    : {model_id}")
    print(f"  Model path  : {getattr(se, 'MODEL_PATH', '?')}")
    print(f"  Log dir     : {getattr(se, 'LOG_DIR', '?')}")
    print(f"  Submit live : {getattr(se, 'SUBMIT_ORDERS', False)}")

    result     = run_trading_cycle()
    decision   = result.get("decision", {})
    action     = decision.get("action", "unknown")

    target_df    = _normalize_target_weights(result.get("target_weights"))
    used_fallback = False

    # ---- Fallback guard ----
    if target_df.empty or "target_weight" not in target_df.columns or target_df["target_weight"].abs().sum() == 0:
        print("\nWARNING: Strategy returned no live target weights.")
        print(f"Creating safe ETF fallback for action: {action}")

        fallback_series          = _fallback_target_from_action(action)
        target_df                = fallback_series.reset_index()
        target_df.columns        = ["symbol", "target_weight"]
        result["target_weights"] = fallback_series
        result["order_plan"]     = _rebuild_order_plan(result, target_df)
        used_fallback            = True

        submit_orders       = bool(getattr(se, "SUBMIT_ORDERS", False))
        allow_fallback      = str(os.environ.get("BRPPO_ALLOW_FALLBACK_ORDERS", "False")).lower() in {"1", "true", "yes"}

        if submit_orders and not allow_fallback:
            print("\nSAFETY BLOCK: Fallback target used — orders NOT submitted.")
            print("Set BRPPO_ALLOW_FALLBACK_ORDERS=True to allow fallback ETF orders.")
            result["submitted_orders"] = pd.DataFrame()
        elif submit_orders and allow_fallback:
            print("\nSubmitting fallback ETF orders (BRPPO_ALLOW_FALLBACK_ORDERS=True).")
            try:
                result["submitted_orders"] = se.submit_orders(result["order_plan"])
            except Exception as exc:
                print("Fallback order submission failed:", repr(exc))
                result["submitted_orders"] = pd.DataFrame()

        try:
            se.log_outputs(
                decision=result.get("decision", {}),
                target_weights=result["target_weights"],
                positions=result.get("positions", pd.DataFrame()),
                order_plan=result.get("order_plan", pd.DataFrame()),
                submitted_orders=result.get("submitted_orders", pd.DataFrame()),
                account_value=float(decision.get("account_value", getattr(se, "DEFAULT_ACCOUNT_VALUE", 100_000.0))),
            )
        except Exception as exc:
            print("Could not re-log fallback output:", repr(exc))

    # ---- Final display ----
    target_df       = _normalize_target_weights(result.get("target_weights"))
    order_plan = result.get("order_plan"); order_plan = pd.DataFrame() if order_plan is None else order_plan
    submitted_orders = result.get("submitted_orders"); submitted_orders = pd.DataFrame() if submitted_orders is None else submitted_orders

    print("\n" + "=" * 70)
    print("BR-PPO ALPACA PAPER TRADER COMPLETE")
    print("=" * 70)

    print("\nDecision:")
    for k, v in decision.items():
        print(f"  {k}: {v}")

    if used_fallback:
        print("\nNOTE: Fallback ETF target was used (live basket was empty).")

    print("\nTop target weights:")
    if not target_df.empty and "target_weight" in target_df.columns:
        print(target_df.sort_values("target_weight", ascending=False).head(50).to_string(index=False))
    else:
        print("No target weights available.")

    print("\nPlanned orders:")
    if isinstance(order_plan, pd.DataFrame) and not order_plan.empty:
        print(order_plan.head(100).to_string(index=False))
    else:
        print("No planned orders.")

    print("\nSubmitted orders:")
    if isinstance(submitted_orders, pd.DataFrame) and not submitted_orders.empty:
        print(submitted_orders.head(100).to_string(index=False))
    else:
        print("No submitted orders.")

    print("\nSafety status:")
    print(f"  BRPPO_SUBMIT_ORDERS : {getattr(se, 'SUBMIT_ORDERS', False)}")
    print(f"  Fallback target used: {used_fallback}")
    print()

    return result


if __name__ == "__main__":
    main()
