import pandas as pd
import numpy as np

def compute_cagr(equity_curve: pd.Series) -> float:
    """Computes Compound Annual Growth Rate."""
    if len(equity_curve) < 2:
        return 0.0
    start_val = equity_curve.iloc[0]
    end_val = equity_curve.iloc[-1]
    n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if n_years <= 0 or start_val <= 0:
        return 0.0
    return float((end_val / start_val) ** (1.0 / n_years) - 1.0)

def compute_max_drawdown(equity_curve: pd.Series) -> tuple:
    """
    Computes Maximum Drawdown and Max Drawdown Duration (in days).
    Returns (max_drawdown_pct, max_drawdown_series).
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    return max_dd, drawdown

def compute_sharpe_ratio(returns: pd.Series, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    """Computes annualized Sharpe Ratio."""
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_daily
    std = excess_returns.std()
    if std <= 1e-8:
        return 0.0
    return float((excess_returns.mean() / std) * np.sqrt(periods_per_year))

def compute_sortino_ratio(returns: pd.Series, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    """Computes annualized Sortino Ratio."""
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_daily
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    if downside_std <= 1e-8:
        return 0.0
    return float((excess_returns.mean() / downside_std) * np.sqrt(periods_per_year))

def compute_alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series, rf_annual: float = 0.02, periods_per_year: int = 252) -> tuple:
    """
    Computes annualized Alpha and Beta vs SPY benchmark via linear regression.
    """
    aligned = pd.concat([strategy_returns.rename("strat"), benchmark_returns.rename("bench")], axis=1).dropna()
    if len(aligned) < 30:
        return 0.0, 1.0
        
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    y = aligned["strat"] - rf_daily
    x = aligned["bench"] - rf_daily
    
    cov = np.cov(x, y)[0, 1]
    var_x = np.var(x)
    
    if var_x <= 1e-8:
        beta = 1.0
    else:
        beta = float(cov / var_x)
        
    alpha_daily = float(y.mean() - beta * x.mean())
    alpha_annual = float((1 + alpha_daily) ** periods_per_year - 1)
    
    return alpha_annual, beta

def compute_performance_metrics(strategy_equity: pd.Series, benchmark_equity: pd.Series, rf_annual: float = 0.02) -> dict:
    """
    Returns comprehensive risk/return performance statistics dictionary.
    """
    strat_ret = strategy_equity.pct_change().dropna()
    bench_ret = benchmark_equity.pct_change().dropna()
    
    strat_cagr = compute_cagr(strategy_equity)
    bench_cagr = compute_cagr(benchmark_equity)
    
    strat_max_dd, strat_dd_series = compute_max_drawdown(strategy_equity)
    bench_max_dd, bench_dd_series = compute_max_drawdown(benchmark_equity)
    
    strat_sharpe = compute_sharpe_ratio(strat_ret, rf_annual=rf_annual)
    bench_sharpe = compute_sharpe_ratio(bench_ret, rf_annual=rf_annual)
    
    strat_sortino = compute_sortino_ratio(strat_ret, rf_annual=rf_annual)
    bench_sortino = compute_sortino_ratio(bench_ret, rf_annual=rf_annual)
    
    alpha, beta = compute_alpha_beta(strat_ret, bench_ret, rf_annual=rf_annual)
    
    strat_vol = float(strat_ret.std() * np.sqrt(252))
    bench_vol = float(bench_ret.std() * np.sqrt(252))
    
    return {
        "CAGR": {"Strategy": strat_cagr, "Benchmark": bench_cagr},
        "Sharpe Ratio": {"Strategy": strat_sharpe, "Benchmark": bench_sharpe},
        "Sortino Ratio": {"Strategy": strat_sortino, "Benchmark": bench_sortino},
        "Max Drawdown": {"Strategy": strat_max_dd, "Benchmark": bench_max_dd},
        "Volatility": {"Strategy": strat_vol, "Benchmark": bench_vol},
        "Alpha": alpha,
        "Beta": beta,
        "Strategy_Drawdown_Series": strat_dd_series,
        "Benchmark_Drawdown_Series": bench_dd_series
    }
