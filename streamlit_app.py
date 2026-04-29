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

LOG_DIR = Path(__file__).parent / "logs"

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


def load_all():
    return {
        "decisions":        load_csv(LOG_DIR / "decisions"      / "decisions.csv"),
        "latest_decision":  load_csv(LOG_DIR / "decisions"      / "latest_decision.csv"),
        "portfolio":        load_csv(LOG_DIR / "portfolio"       / "portfolio.csv"),
        "target_weights":   load_csv(LOG_DIR / "target_weights"  / "latest_target_weights.csv"),
        "tw_history":       load_csv(LOG_DIR / "target_weights"  / "target_weights.csv"),
        "positions":        load_csv(LOG_DIR / "positions"       / "latest_positions.csv"),
        "planned_orders":   load_csv(LOG_DIR / "orders"          / "latest_planned_orders.csv"),
        "submitted_orders": load_csv(LOG_DIR / "orders"          / "latest_submitted_orders.csv"),
        "orders_history":   load_csv(LOG_DIR / "orders"          / "submitted_orders.csv"),
        "signal_history":   load_csv(LOG_DIR / "health"          / "signal_history.csv"),
        "health_status":    _load_health_status(),
    }


def _load_health_status():
    path = LOG_DIR / "health" / "health_status.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


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

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    data = load_all()
    dec  = data["latest_decision"]
    port = data["portfolio"]

    # Last run time
    if not dec.empty and "timestamp_utc" in dec.columns:
        last_run = pd.to_datetime(dec["timestamp_utc"].iloc[-1], utc=True)
        now_utc  = datetime.now(timezone.utc)
        delta_m  = int((now_utc - last_run).total_seconds() / 60)
        st.markdown(f"**Last run:** {last_run.strftime('%b %d, %H:%M UTC')}")
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
st.markdown("*Powered by Proximal Policy Optimization — V10 Allocation Agent*")
st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab_overview, tab_perf, tab_portfolio, tab_orders, tab_history, tab_health = st.tabs([
    "🏠 Overview",
    "📊 Performance",
    "🎯 Portfolio",
    "📋 Orders",
    "📜 History",
    "🧠 Model Health",
])


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
    port     = data["portfolio"]
    decisions = data["decisions"]

    st.markdown('<div class="section-header">Portfolio Value Over Time</div>', unsafe_allow_html=True)

    if port.empty:
        st.info("No portfolio history yet.")
    else:
        port_plot = port.copy()
        if "timestamp_utc" in port_plot.columns:
            port_plot["timestamp_utc"] = pd.to_datetime(port_plot["timestamp_utc"], utc=True)
            port_plot = port_plot.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")

        # Compute daily returns
        if len(port_plot) > 1:
            port_plot["daily_return"] = port_plot["portfolio_value"].pct_change()
            port_plot["cum_return"]   = (1 + port_plot["daily_return"].fillna(0)).cumprod() - 1

            # Running drawdown
            peak = port_plot["portfolio_value"].cummax()
            port_plot["drawdown"] = port_plot["portfolio_value"] / peak - 1

        # ── Performance metrics ──
        if len(port_plot) > 1:
            start_v   = port_plot["portfolio_value"].iloc[0]
            end_v     = port_plot["portfolio_value"].iloc[-1]
            total_ret = (end_v - start_v) / start_v * 100
            max_dd    = port_plot["drawdown"].min() * 100
            n_days    = (port_plot["timestamp_utc"].iloc[-1] - port_plot["timestamp_utc"].iloc[0]).days
            ann_ret   = (((end_v / start_v) ** (365 / max(n_days, 1))) - 1) * 100 if n_days > 0 else 0
            vol       = port_plot["daily_return"].std() * np.sqrt(252) * 100 if "daily_return" in port_plot.columns else 0
            sharpe    = ann_ret / vol if vol > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                sign = "pos" if total_ret >= 0 else "neg"
                metric_card("Total Return", f"{'+' if total_ret>=0 else ''}{total_ret:.2f}%",
                            delta_sign=sign)
            with m2:
                metric_card("Ann. Return (proj)", f"{ann_ret:+.2f}%",
                            delta=f"Over {n_days} days",
                            delta_sign="pos" if ann_ret >= 0 else "neg")
            with m3:
                metric_card("Max Drawdown", f"{max_dd:.2f}%", delta_sign="neg")
            with m4:
                metric_card("Sharpe (proj)", f"{sharpe:.2f}")

            st.markdown("")

        # ── Portfolio value chart ──
        fig = go.Figure()
        x_vals = port_plot["timestamp_utc"] if "timestamp_utc" in port_plot.columns else port_plot.index
        y_vals = port_plot["portfolio_value"]
        color  = PALETTE["green"] if y_vals.iloc[-1] >= y_vals.iloc[0] else PALETTE["red"]

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=hex_to_rgba(color, 0.09),
            marker=dict(size=4, color=color),
            name="Portfolio Value",
            hovertemplate="$%{y:,.2f}<br>%{x}<extra></extra>",
        ))

        # Action annotations
        if not decisions.empty and "action" in decisions.columns and "timestamp_utc" in decisions.columns:
            decisions_plot = decisions.copy()
            decisions_plot["timestamp_utc"] = pd.to_datetime(decisions_plot["timestamp_utc"], utc=True)
            for _, row in decisions_plot.tail(30).iterrows():
                fig.add_vline(
                    x=row["timestamp_utc"],
                    line_dash="dot",
                    line_color=ACTION_COLORS.get(str(row["action"]).lower(), "#8892a4"),
                    line_width=1,
                    opacity=0.5,
                )

        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=16, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color=PALETTE["muted"]),
            yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"],
                       tickprefix="$", tickformat=",.0f"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Drawdown chart ──
        if "drawdown" in port_plot.columns and len(port_plot) > 1:
            st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=x_vals,
                y=port_plot["drawdown"] * 100,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PALETTE["red"], width=1.5),
                fillcolor=f"{PALETTE['red']}25",
                hovertemplate="%{y:.2f}%<br>%{x}<extra></extra>",
            ))
            fig2.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color=PALETTE["muted"]),
                yaxis=dict(showgrid=True, gridcolor="#1e2535", color=PALETTE["muted"],
                           ticksuffix="%"),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Action history ──
    if not decisions.empty and "action" in decisions.columns:
        st.markdown('<div class="section-header">Action Distribution</div>', unsafe_allow_html=True)
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
    overall = health.get("overall_status", "unknown")
    status_colors = {"healthy": "#00d4aa", "warning": "#ffd166", "degraded": "#ff4b6e", "unknown": "#8892a4"}
    status_icons  = {"healthy": "✅", "warning": "⚠️", "degraded": "🚨", "unknown": "❓"}
    sc = status_colors.get(overall, "#8892a4")
    si = status_icons.get(overall, "❓")

    st.markdown(f"""
    <div style="background:{sc}22; border:2px solid {sc}; border-radius:12px; padding:20px 24px; margin-bottom:20px;">
        <span style="font-size:24px; font-weight:700; color:{sc};">{si} Model Status: {overall.upper()}</span>
        <div style="color:#8892a4; font-size:13px; margin-top:6px;">
            Last checked: {health.get('computed_at', 'Never')[:19].replace('T',' ')} UTC
            &nbsp;|&nbsp; Lookback: {health.get('lookback_days', 63)} days
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Alerts ──
    alerts = health.get("alerts", [])
    if alerts:
        for alert in alerts:
            st.warning(alert)
    elif overall == "healthy":
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
            spy_data = yf.download("SPY", start=port_plot.index[0].strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
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

