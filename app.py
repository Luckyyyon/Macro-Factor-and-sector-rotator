import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
from plotly.subplots import make_subplots
import datetime

from src.data_ingestion import load_combined_dataset, DEFAULT_SECTOR_ETFS
from src.factor_model import compute_all_sector_betas, compute_macro_factor_changes
from src.regime_detector import classify_macro_regime, REGIMES
from src.strategy_engine import run_backtest
from src.performance import compute_performance_metrics

st.set_page_config(
    page_title="Macro Factor Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished quantitative dashboard UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2a2e39;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00e676;
    }
    .metric-label {
        font-size: 13px;
        color: #8a919e;
        margin-bottom: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
        background-color: #1e222d;
    }
</style>
""", unsafe_allow_html=True)

st.title("Cross-Asset Macro Factor & Sector Rotation Engine")
st.caption("Quantitative Macro Regime Detection | Dynamic Rolling Factor Betas | Backtest vs SPY Benchmark")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")

fred_key = st.sidebar.text_input("FRED API Key (Optional)", type="password", help="Leave blank to use direct web fallback endpoint.")

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    start_date = st.date_input("Start Date", datetime.date(2012, 1, 1))
with col_s2:
    end_date = st.date_input("End Date", datetime.date(2025, 12, 31))

initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000.0, step=10000.0)
slippage_bps = st.sidebar.slider("Slippage per Trade (bps)", min_value=0, max_value=50, value=10, help="10 bps = 0.10% round-trip transaction cost")
rolling_window = st.sidebar.slider("OLS Rolling Beta Window (Days)", min_value=63, max_value=504, value=252, step=21)

selected_sectors = st.sidebar.multiselect(
    "Sector Universe",
    options=["XLE", "XLF", "XLK", "XLU", "XLV", "XLI", "XLY", "XLP", "XLC", "XLB", "XLRE"],
    default=["XLE", "XLF", "XLK", "XLU", "XLV", "XLI", "XLY", "XLP", "XLB"]
)

run_button = st.sidebar.button("Run Macro Backtest", type="primary", use_container_width=True)

# Data Caching
@st.cache_data(ttl=3600, show_spinner="Ingesting FRED Macro & Sector ETF Data...")
def load_data(s_date, e_date, api_key):
    s_str = s_date.strftime("%Y-%m-%d")
    e_str = e_date.strftime("%Y-%m-%d") if e_date else None
    return load_combined_dataset(start_date=s_str, end_date=e_str, fred_api_key=api_key)

try:
    combined_df, prices_df, macro_df = load_data(start_date, end_date, fred_key)
except Exception as e:
    st.error(f"Failed to load market data: {e}")
    st.stop()

# Ensure benchmark is present
if "SPY" not in prices_df.columns:
    st.error("SPY benchmark price missing.")
    st.stop()

# Filter sector universe
available_sectors = [c for c in selected_sectors if c in prices_df.columns]
active_prices = prices_df[["SPY"] + available_sectors]

# Run Backtest
with st.spinner("Running Regime Classification & Backtest..."):
    strat_series, bench_series, weights_df, regime_series, metrics = run_backtest(
        active_prices,
        macro_df,
        initial_capital=initial_capital,
        slippage=slippage_bps / 10000.0,
        benchmark_col="SPY"
    )

# Top KPI Summary Cards
st.subheader("Key Performance Metrics")
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

def format_kpi(val, is_pct=True):
    if is_pct:
        return f"{val * 100:.2f}%"
    return f"{val:.2f}"

with kpi1:
    st.metric("Strategy CAGR", format_kpi(metrics["CAGR"]["Strategy"]), delta=f"{(metrics['CAGR']['Strategy'] - metrics['CAGR']['Benchmark'])*100:+.2f}% vs SPY")
with kpi2:
    st.metric("Benchmark CAGR", format_kpi(metrics["CAGR"]["Benchmark"]))
with kpi3:
    st.metric("Strategy Sharpe", format_kpi(metrics["Sharpe Ratio"]["Strategy"], False), delta=f"{metrics['Sharpe Ratio']['Strategy'] - metrics['Sharpe Ratio']['Benchmark']:+.2f} vs SPY")
with kpi4:
    st.metric("Max Drawdown", format_kpi(metrics["Max Drawdown"]["Strategy"]), delta=f"{(metrics['Max Drawdown']['Strategy'] - metrics['Max Drawdown']['Benchmark'])*100:+.2f}% vs SPY")
with kpi5:
    st.metric("Alpha (Annual)", format_kpi(metrics["Alpha"]))
with kpi6:
    st.metric("Beta vs SPY", format_kpi(metrics["Beta"], False))

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Equity Curve & Allocation",
    "Dynamic Rolling Betas",
    "Macro Regimes & Transitions",
    "Performance & Drawdown Profile"
])

# Tab 1: Equity Curve & Allocation
with tab1:
    st.subheader("Cumulative Portfolio Performance vs Benchmark")
    
    # Plotly Cumulative Return Comparison
    df_perf = pd.DataFrame({
        "Strategy": strat_series,
        "SPY Benchmark": bench_series
    })
    
    fig_equity = px.line(
        df_perf,
        x=df_perf.index,
        y=["Strategy", "SPY Benchmark"],
        labels={"value": "Portfolio Value ($)", "variable": "Portfolio"},
        color_discrete_map={"Strategy": "#00e676", "SPY Benchmark": "#29b6f6"}
    )
    fig_equity.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Historical Asset Weights Stacked Area Chart
    st.subheader("Dynamic Sector Weight Allocation History")
    fig_weights = px.area(
        weights_df,
        x=weights_df.index,
        y=weights_df.columns,
        labels={"value": "Allocation Weight", "variable": "Sector ETF"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_weights.update_layout(
        template="plotly_dark",
        height=350,
        hovermode="x unified",
        yaxis=dict(tickformat=".0%")
    )
    st.plotly_chart(fig_weights, use_container_width=True)

# Tab 2: Dynamic Rolling Betas
with tab2:
    st.subheader(f"Rolling {rolling_window}-Day OLS Beta Exposures to Macro Factors")
    
    with st.spinner("Calculating Rolling OLS Regressions..."):
        all_betas = compute_all_sector_betas(active_prices, macro_df, window=rolling_window)
        
    select_etf = st.selectbox("Select Sector ETF to Analyze", options=available_sectors, index=0)
    
    if select_etf in all_betas and not all_betas[select_etf].empty:
        beta_df = all_betas[select_etf]
        
        # Plot Time-Series of Betas for selected ETF
        beta_cols = [c for c in beta_df.columns if c.startswith("Beta_")]
        
        fig_betas = px.line(
            beta_df,
            x=beta_df.index,
            y=beta_cols,
            title=f"{select_etf} Rolling Macro Factor Betas",
            labels={"value": "Beta Coefficient", "variable": "Macro Factor"},
            template="plotly_dark"
        )
        fig_betas.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_betas, use_container_width=True)
        
        # R-Squared chart
        fig_r2 = px.area(
            beta_df,
            x=beta_df.index,
            y="R_squared",
            title=f"{select_etf} Macro Factor Model R-Squared (Explanatory Power)",
            template="plotly_dark",
            color_discrete_sequence=["#ab47bc"]
        )
        fig_r2.update_layout(height=250, yaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # Heatmap of Current/Latest Factor Betas Across All Sectors
        st.subheader("Current Sector Beta Exposure Heatmap")
        latest_betas = []
        for ticker, b_df in all_betas.items():
            if not b_df.empty:
                last_row = b_df.iloc[-1].to_dict()
                last_row["Sector"] = ticker
                latest_betas.append(last_row)
                
        if latest_betas:
            heatmap_df = pd.DataFrame(latest_betas).set_index("Sector")
            heatmap_cols = [c for c in heatmap_df.columns if c.startswith("Beta_")]
            
            fig_heatmap = px.imshow(
                heatmap_df[heatmap_cols],
                labels=dict(x="Macro Factor", y="Sector ETF", color="Beta"),
                x=[c.replace("Beta_", "") for c in heatmap_cols],
                y=heatmap_df.index,
                color_continuous_scale="RdBu_r",
                text_auto=".2f",
                aspect="auto",
                template="plotly_dark"
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Insufficient historical data window for rolling OLS calculations.")

# Tab 3: Macro Regimes & Transitions
with tab3:
    st.subheader("Macroeconomic Regime Classification History")
    
    regime_colors = {
        "EXPANSION": "#00c853",
        "SLOWDOWN": "#ffab00",
        "STAGFLATION": "#ff3d00",
        "CONTRACTION": "#d500f9"
    }
    
    # Plot Macro Indicators + Regime Banner
    fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                              subplot_titles=("10Y Treasury Yield & 10Y-2Y Slope (%)", "CPI Inflation Index", "Detected Macro Regime"))
    
    if "10Y_Yield" in macro_df.columns:
        fig_macro.add_trace(gg.Scatter(x=macro_df.index, y=macro_df["10Y_Yield"], name="10Y Yield", line=dict(color="#ff9100")), row=1, col=1)
    if "Yield_Curve_10Y2Y" in macro_df.columns:
        fig_macro.add_trace(gg.Scatter(x=macro_df.index, y=macro_df["Yield_Curve_10Y2Y"], name="10Y-2Y Curve Slope", line=dict(color="#29b6f6")), row=1, col=1)
        
    if "CPI_Inflation" in macro_df.columns:
        fig_macro.add_trace(gg.Scatter(x=macro_df.index, y=macro_df["CPI_Inflation"], name="CPI Index", line=dict(color="#ab47bc")), row=2, col=1)
        
    # Map regime series to numerical codes for plotting
    reg_map = {"EXPANSION": 1, "SLOWDOWN": 2, "STAGFLATION": 3, "CONTRACTION": 4}
    reg_num = regime_series.map(reg_map)
    fig_macro.add_trace(gg.Scatter(x=regime_series.index, y=reg_num, mode="markers+lines", name="Macro Regime",
                                   line=dict(color="#00e676", width=1.5)), row=3, col=1)
    
    fig_macro.update_layout(height=650, template="plotly_dark", showlegend=True)
    fig_macro.update_yaxes(tickvals=[1, 2, 3, 4], ticktext=["EXPANSION", "SLOWDOWN", "STAGFLATION", "CONTRACTION"], row=3, col=1)
    st.plotly_chart(fig_macro, use_container_width=True)
    
    # Regime Distribution Breakdown
    st.subheader("Historical Regime Distribution")
    reg_counts = regime_series.value_counts(normalize=True).reset_index()
    reg_counts.columns = ["Regime", "Percentage"]
    reg_counts["Percentage"] *= 100
    
    fig_pie = px.pie(
        reg_counts,
        names="Regime",
        values="Percentage",
        color="Regime",
        color_discrete_map=regime_colors,
        template="plotly_dark",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Tab 4: Performance & Drawdown Profile
with tab4:
    st.subheader("Drawdown Profile Comparison")
    
    dd_df = pd.DataFrame({
        "Strategy Drawdown": metrics["Strategy_Drawdown_Series"],
        "Benchmark Drawdown": metrics["Benchmark_Drawdown_Series"]
    })
    
    fig_dd = px.line(
        dd_df,
        x=dd_df.index,
        y=["Strategy Drawdown", "Benchmark Drawdown"],
        labels={"value": "Drawdown", "variable": "Series"},
        color_discrete_map={"Strategy Drawdown": "#ff5252", "Benchmark Drawdown": "#ffb74d"},
        template="plotly_dark"
    )
    fig_dd.update_layout(height=400, yaxis=dict(tickformat=".1%"), hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Detailed Risk & Return Comparison Table
    st.subheader("Risk & Return Metrics Breakdown")
    
    summary_data = {
        "Metric": ["CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Annual Volatility"],
        "Strategy": [
            f"{metrics['CAGR']['Strategy']*100:.2f}%",
            f"{metrics['Sharpe Ratio']['Strategy']:.2f}",
            f"{metrics['Sortino Ratio']['Strategy']:.2f}",
            f"{metrics['Max Drawdown']['Strategy']*100:.2f}%",
            f"{metrics['Volatility']['Strategy']*100:.2f}%"
        ],
        "SPY Benchmark": [
            f"{metrics['CAGR']['Benchmark']*100:.2f}%",
            f"{metrics['Sharpe Ratio']['Benchmark']:.2f}",
            f"{metrics['Sortino Ratio']['Benchmark']:.2f}",
            f"{metrics['Max Drawdown']['Benchmark']*100:.2f}%",
            f"{metrics['Volatility']['Benchmark']*100:.2f}%"
        ]
    }
    st.table(pd.DataFrame(summary_data))
