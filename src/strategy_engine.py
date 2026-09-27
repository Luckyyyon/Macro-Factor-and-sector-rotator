import pandas as pd
import numpy as np
import logging
from src.regime_detector import classify_macro_regime, get_target_weights_for_regime
from src.performance import compute_performance_metrics

def run_backtest(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    slippage: float = 0.0010,  # 0.10% round-trip slippage
    benchmark_col: str = "SPY"
) -> tuple:
    """
    Backtests regime-switching sector rotation strategy with monthly rebalancing on 1st trading day.
    Guarantees Zero Lookahead Bias:
    - Rebalance signal for Month M uses macro data & regime up to the LAST trading day of Month M-1 (t-1).
    - Allocation weights strictly sum to 1.0.
    
    Returns:
    - strategy_equity (pd.Series)
    - benchmark_equity (pd.Series)
    - weights_history (pd.DataFrame)
    - regime_history (pd.Series)
    - metrics (dict)
    """
    # Filter tickers
    sector_tickers = [c for c in prices_df.columns if c != benchmark_col]
    
    # Classify daily regimes
    daily_regimes = classify_macro_regime(macro_df)
    
    # Identify month end / start dates for rebalancing
    # We rebalance on the FIRST trading day of each month using regime determined at the END of the previous month
    prices_df = prices_df.copy().sort_index()
    returns_df = prices_df.pct_change().fillna(0.0)
    
    dates = prices_df.index
    
    # Resample to monthly to find 1st trading day of each month
    prices_df["YearMonth"] = prices_df.index.to_period("M")
    first_days_of_month = prices_df.groupby("YearMonth").head(1).index
    prices_df = prices_df.drop(columns=["YearMonth"])
    
    current_weights = {t: 1.0 / len(sector_tickers) for t in sector_tickers} # start equal weight
    
    strategy_values = [initial_capital]
    benchmark_values = [initial_capital]
    
    weights_records = []
    regime_records = []
    
    # Track current regime
    current_regime = daily_regimes.iloc[0]
    
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        
        # Check if today is a rebalance day (1st trading day of month)
        if curr_date in first_days_of_month:
            # ZERO LOOKAHEAD BIAS: signal determined at prev_date (t-1)
            regime_at_t_minus_1 = daily_regimes.loc[prev_date]
            target_w = get_target_weights_for_regime(regime_at_t_minus_1, available_sectors=sector_tickers)
            
            # Compute turn-over for slippage calculation
            turnover = 0.0
            for t in sector_tickers:
                old_w = current_weights.get(t, 0.0)
                new_w = target_w.get(t, 0.0)
                turnover += abs(new_w - old_w)
                
            current_weights = target_w
            current_regime = regime_at_t_minus_1
            
            # Apply transaction slippage on rebalance day
            slippage_cost = strategy_values[-1] * turnover * slippage
            strat_val_start = strategy_values[-1] - slippage_cost
        else:
            strat_val_start = strategy_values[-1]
            
        # Record weights & regime for curr_date
        weights_row = {"date": curr_date}
        for t in sector_tickers:
            weights_row[t] = current_weights.get(t, 0.0)
        weights_records.append(weights_row)
        regime_records.append({"date": curr_date, "regime": current_regime})
        
        # Compute daily return for strategy based on current_weights
        daily_strat_ret = 0.0
        for t, w in current_weights.items():
            if t in returns_df.columns:
                daily_strat_ret += w * returns_df.loc[curr_date, t]
                
        new_strat_val = strat_val_start * (1.0 + daily_strat_ret)
        strategy_values.append(new_strat_val)
        
        # Benchmark daily return
        bench_ret = returns_df.loc[curr_date, benchmark_col] if benchmark_col in returns_df.columns else 0.0
        new_bench_val = benchmark_values[-1] * (1.0 + bench_ret)
        benchmark_values.append(new_bench_val)
        
    strat_series = pd.Series(strategy_values, index=dates, name="Strategy")
    bench_series = pd.Series(benchmark_values, index=dates, name="Benchmark (SPY)")
    
    weights_df = pd.DataFrame(weights_records).set_index("date")
    regime_series = pd.DataFrame(regime_records).set_index("date")["regime"]
    
    # Calculate average 10Y yield as Risk-Free rate proxy
    rf_mean = 0.02
    if "10Y_Yield" in macro_df.columns:
        rf_mean = float(macro_df["10Y_Yield"].mean() / 100.0)
        
    metrics = compute_performance_metrics(strat_series, bench_series, rf_annual=rf_mean)
    
    return strat_series, bench_series, weights_df, regime_series, metrics
