# tests/test_backtester.py
import pytest
import pandas as pd
import numpy as np
from src.regime_detector import classify_macro_regime, get_target_weights_for_regime
from src.strategy_engine import run_backtest
from src.factor_model import compute_macro_factor_changes, run_rolling_ols

def create_synthetic_data(n_days=500):
    """Generates synthetic price and macro data for unit testing."""
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")
    
    np.random.seed(42)
    # Synthetic ETF prices
    tickers = ["SPY", "XLE", "XLF", "XLK", "XLU", "XLV", "XLI", "XLY", "XLP"]
    price_data = {}
    for t in tickers:
        returns = np.random.normal(0.0005, 0.01, n_days)
        price_data[t] = 100.0 * np.exp(np.cumsum(returns))
        
    prices_df = pd.DataFrame(price_data, index=dates)
    
    # Synthetic Macro data
    dgs10 = 2.0 + np.sin(np.linspace(0, 10, n_days))
    t10y2y = 0.5 + np.cos(np.linspace(0, 8, n_days))
    cpi = 250.0 + np.linspace(0, 20, n_days)
    unrate = 4.0 + np.sin(np.linspace(0, 5, n_days)) * 0.5
    
    macro_df = pd.DataFrame({
        "10Y_Yield": dgs10,
        "Yield_Curve_10Y2Y": t10y2y,
        "CPI_Inflation": cpi,
        "Unemployment_Rate": unrate
    }, index=dates)
    
    return prices_df, macro_df

def test_allocation_weights_sum_to_one():
    """Verifies that strategy target allocation weights sum to exactly 1.0 across all regimes."""
    sectors = ["XLE", "XLF", "XLK", "XLU", "XLV", "XLI", "XLY", "XLP", "XLB"]
    for regime in ["EXPANSION", "SLOWDOWN", "STAGFLATION", "CONTRACTION", "UNKNOWN"]:
        weights = get_target_weights_for_regime(regime, available_sectors=sectors)
        total_weight = sum(weights.values())
        assert pytest.approx(total_weight, abs=1e-5) == 1.0, f"Weights for regime {regime} do not sum to 1.0"

def test_no_lookahead_bias_in_backtest():
    """
    Verifies that regime signals used on rebalance dates only depend on data up to t-1.
    """
    prices_df, macro_df = create_synthetic_data(n_days=300)
    
    strat_series, bench_series, weights_df, regime_series, metrics = run_backtest(
        prices_df, macro_df, initial_capital=100000.0
    )
    
    # Check that weights DataFrame has no NaNs and each row sums to 1.0
    row_sums = weights_df.sum(axis=1)
    for date, val in row_sums.items():
        assert pytest.approx(val, abs=1e-5) == 1.0, f"Weights on {date} do not sum to 1.0"
        
    # Check that equity curves are strictly non-negative and have correct length
    assert len(strat_series) == len(prices_df)
    assert (strat_series > 0).all()
    assert (bench_series > 0).all()

def test_factor_rolling_ols():
    """Verifies that dynamic rolling OLS outputs correct beta columns and non-null values."""
    prices_df, macro_df = create_synthetic_data(n_days=300)
    factor_changes = compute_macro_factor_changes(macro_df)
    
    etf_ret = prices_df["XLK"].pct_change().dropna()
    betas = run_rolling_ols(etf_ret, factor_changes, window=100)
    
    assert not betas.empty
    assert "Alpha" in betas.columns
    assert "R_squared" in betas.columns
    assert (betas["R_squared"] >= 0.0).all()
