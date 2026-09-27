import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging

def compute_macro_factor_changes(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily/monthly factor changes for macro variables to use as regressor inputs:
    - d_10Y: Daily change in 10-Yr Treasury yield (percentage point change)
    - Slope: 10Y - 2Y yield curve level (or daily change)
    - d_CPI: 1-month or 12-month YoY change in CPI Inflation
    """
    factor_df = pd.DataFrame(index=macro_df.index)
    
    if "10Y_Yield" in macro_df.columns:
        factor_df["d_10Y"] = macro_df["10Y_Yield"].diff()
        
    if "Yield_Curve_10Y2Y" in macro_df.columns:
        factor_df["Slope_10Y2Y"] = macro_df["Yield_Curve_10Y2Y"]
        factor_df["d_Slope"] = macro_df["Yield_Curve_10Y2Y"].diff()
        
    if "CPI_Inflation" in macro_df.columns:
        # MoM or YoY CPI change
        factor_df["CPI_YoY"] = macro_df["CPI_Inflation"].pct_change(252) * 100
        factor_df["d_CPI"] = macro_df["CPI_Inflation"].pct_change(21) * 100
        
    if "Unemployment_Rate" in macro_df.columns:
        factor_df["d_UNRATE"] = macro_df["Unemployment_Rate"].diff()

    factor_df = factor_df.fillna(0.0)
    return factor_df

def run_rolling_ols(etf_returns: pd.Series, factor_changes: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Runs a rolling multivariate OLS regression of daily ETF returns against macro factor changes.
    window: 252 trading days (~12 months).
    Returns DataFrame of dynamic beta coefficients (plus R-squared) over time.
    """
    aligned = pd.concat([etf_returns.rename("etf_ret"), factor_changes], axis=1).dropna()
    
    if len(aligned) < window:
        return pd.DataFrame()

    feature_cols = [c for c in factor_changes.columns if c in aligned.columns]
    
    dates = aligned.index
    betas_list = []
    
    # Pre-allocate array for speed
    Y = aligned["etf_ret"].values
    X = aligned[feature_cols].values
    
    for i in range(window, len(aligned) + 1):
        sub_Y = Y[i-window:i]
        sub_X = X[i-window:i]
        
        # Add constant intercept
        sub_X_const = np.column_stack([np.ones(window), sub_X])
        
        curr_date = dates[i-1]
        
        try:
            # Solve OLS via np.linalg.lstsq
            coefs, residuals, rank, s = np.linalg.lstsq(sub_X_const, sub_Y, rcond=None)
            
            # Calculate R-squared
            ss_tot = np.sum((sub_Y - np.mean(sub_Y))**2)
            ss_res = np.sum((sub_Y - sub_X_const @ coefs)**2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            row = {"date": curr_date, "Alpha": coefs[0]}
            for idx, col in enumerate(feature_cols):
                row[f"Beta_{col}"] = coefs[idx + 1]
            row["R_squared"] = max(0.0, r2)
            
            betas_list.append(row)
        except Exception:
            pass
            
    res_df = pd.DataFrame(betas_list).set_index("date")
    return res_df

def compute_all_sector_betas(prices_df: pd.DataFrame, macro_df: pd.DataFrame, window: int = 252) -> dict:
    """
    Computes rolling factor betas for all sector ETFs in prices_df.
    Returns a dictionary mapping ticker -> rolling betas DataFrame.
    """
    returns_df = prices_df.pct_change().dropna()
    factor_changes = compute_macro_factor_changes(macro_df)
    
    all_betas = {}
    sector_tickers = [c for c in prices_df.columns if c != "SPY"]
    
    for ticker in sector_tickers:
        if ticker in returns_df.columns:
            betas = run_rolling_ols(returns_df[ticker], factor_changes, window=window)
            all_betas[ticker] = betas
            
    return all_betas
