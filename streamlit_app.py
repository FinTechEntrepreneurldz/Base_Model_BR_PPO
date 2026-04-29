"""
BR-PPO Alpaca Paper Trading Dashboard
Streamlit Cloud deployment — reads committed log CSVs from the repo.
"""

import streamlit as st
import pandas as pd
import numpy as np
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

tab_overview, tab_perf, tab_portfolio, tab_orders, tab_history = st.tabs([
    "🏠 Overview",
    "📊 Performance",
    "🎯 Portfolio",
    "📋 Orders",
    "📜 History",
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
        metric_card("Today's Action", action)

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
                fillcolor=f"{line_color}18",
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
            fill="tozeroy", fillcolor=f"{color}18",
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
