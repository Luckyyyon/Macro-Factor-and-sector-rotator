import pandas as pd
import numpy as np
import logging

# Define canonical macro regimes
REGIMES = {
    "EXPANSION": "Expansion (Rising Yield Curve / Low/Falling Inflation)",
    "SLOWDOWN": "Slowdown (Flattening/Inverted Curve / Late Cycle)",
    "STAGFLATION": "Stagflation (Rising Inflation / Flattening Curve)",
    "CONTRACTION": "Contraction (Inverted Curve / Falling Inflation / Recession Warning)"
}

def classify_macro_regime(macro_df: pd.DataFrame, lookback_months: int = 3) -> pd.Series:
    """
    Classifies monthly macro regimes strictly using data available up to time t (no lookahead bias).
    
    Regime Matrix Rules based on 3-month momentum of leading indicators:
    - Yield Curve Slope (10Y - 2Y) & 10Y Yield trend
    - CPI Inflation 3-month trend (or YoY momentum)
    
    Regimes:
    1. EXPANSION: Curve Steepening / Healthy + Inflation moderate or falling
    2. SLOWDOWN: Rates High/Rising + Inflation High/Rising + Curve Flattening
    3. STAGFLATION: Inflation Accelerating + Growth/Yield Curve Flattening/Inverted
    4. CONTRACTION: Inverted Yield Curve (T10Y2Y < 0 or sharply flattening) + Falling Inflation / Recession
    """
    df = macro_df.copy()
    
    # 3-month (approx 63 trading days) momentum / change indicators
    if "Yield_Curve_10Y2Y" in df.columns:
        curve_slope = df["Yield_Curve_10Y2Y"]
        curve_change_3m = df["Yield_Curve_10Y2Y"].diff(63)
    else:
        curve_slope = pd.Series(0.5, index=df.index)
        curve_change_3m = pd.Series(0.0, index=df.index)
        
    if "CPI_Inflation" in df.columns:
        cpi_mom_3m = df["CPI_Inflation"].pct_change(63)
    else:
        cpi_mom_3m = pd.Series(0.0, index=df.index)
        
    if "10Y_Yield" in df.columns:
        yield_change_3m = df["10Y_Yield"].diff(63)
    else:
        yield_change_3m = pd.Series(0.0, index=df.index)
        
    regime_series = pd.Series(index=df.index, dtype=str)
    
    for i in range(len(df)):
        c_slope = curve_slope.iloc[i]
        c_change = curve_change_3m.iloc[i]
        cpi_trend = cpi_mom_3m.iloc[i]
        y_change = yield_change_3m.iloc[i]
        
        # Rule set
        # Check for Yield Curve Inversion or Deep Flattening (Recession / Contraction Risk)
        if c_slope < 0.0 or (c_slope < 0.25 and c_change < -0.2):
            if cpi_trend > 0.01: # High/Rising Inflation with inverted curve -> Stagflation
                regime_series.iloc[i] = "STAGFLATION"
            else:
                regime_series.iloc[i] = "CONTRACTION"
        else:
            # Non-inverted curve
            if cpi_trend > 0.008 or (cpi_trend > 0.004 and y_change > 0.25):
                # Rising Inflation
                if c_change < 0 or y_change > 0.3:
                    regime_series.iloc[i] = "SLOWDOWN"
                else:
                    regime_series.iloc[i] = "STAGFLATION"
            else:
                # Moderate/Falling Inflation + Positive Slope
                if c_change >= -0.1:
                    regime_series.iloc[i] = "EXPANSION"
                else:
                    regime_series.iloc[i] = "SLOWDOWN"
                    
    regime_series = regime_series.fillna("EXPANSION")
    return regime_series

# Target Sector Allocations per Regime based on empirical financial theory & macro playbooks
REGIME_TARGET_ALLOCATIONS = {
    "EXPANSION": {
        "XLK": 0.30,  # Tech
        "XLY": 0.25,  # Consumer Discretionary
        "XLI": 0.25,  # Industrials
        "XLF": 0.20,  # Financials
    },
    "SLOWDOWN": {
        "XLF": 0.30,  # Financials
        "XLE": 0.25,  # Energy
        "XLP": 0.25,  # Consumer Staples
        "XLV": 0.20,  # Health Care
    },
    "STAGFLATION": {
        "XLE": 0.35,  # Energy
        "XLB": 0.25,  # Materials
        "XLP": 0.20,  # Consumer Staples
        "XLU": 0.20,  # Utilities
    },
    "CONTRACTION": {
        "XLU": 0.35,  # Utilities
        "XLV": 0.35,  # Health Care
        "XLP": 0.30,  # Consumer Staples
    }
}

def get_target_weights_for_regime(regime: str, available_sectors: list = None) -> dict:
    """
    Returns normalized weight allocation vector summing to 1.0 for a given regime.
    If some target sector is not in available_sectors, renormalizes across remaining.
    """
    weights = REGIME_TARGET_ALLOCATIONS.get(regime, REGIME_TARGET_ALLOCATIONS["EXPANSION"]).copy()
    
    if available_sectors:
        weights = {k: v for k, v in weights.items() if k in available_sectors}
        
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    else:
        # Equal weight as fallback
        if available_sectors:
            eq_w = 1.0 / len(available_sectors)
            weights = {k: eq_w for k in available_sectors}
            
    return weights
