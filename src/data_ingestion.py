import os
import pandas as pd
import numpy as np
import yfinance as yf
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Default SPDR Sector ETFs + SPY Benchmark
DEFAULT_SECTOR_ETFS = [
    "XLE",   # Energy
    "XLF",   # Financials
    "XLK",   # Technology
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "LC",    # (Replaced by XLC in standard lists, we will include XLC and XLB/XME)
    "XLC",   # Communication Services
    "XLB",   # Materials
    "XLRE",  # Real Estate
]

FRED_SERIES_MAP = {
    "DGS10": "10Y_Yield",           # 10-Year Treasury Constant Maturity Rate
    "T10Y2Y": "Yield_Curve_10Y2Y",  # 10-Year Treasury Constant Maturity Minus 2-Year Constant Maturity
    "CPIAUCSL": "CPI_Inflation",    # Consumer Price Index for All Urban Consumers: All Items
    "UNRATE": "Unemployment_Rate"   # Unemployment Rate
}

def fetch_fred_csv_fallback(series_id: str) -> pd.Series:
    """
    Downloads a series from FRED via direct CSV URL fallback.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]
        # Drop non-numeric entries like '.'
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df = df.dropna(subset=['observation_date']).set_index('observation_date')[series_id]
        df = df.sort_index()
        return df
    except Exception as e:
        logging.error(f"Error fetching FRED series {series_id} via CSV URL: {e}")
        return pd.Series(dtype=float)

def fetch_macro_data(api_key: str = None, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Fetches macroeconomic series (DGS10, T10Y2Y, CPIAUCSL, UNRATE) from FRED API or direct CSV fallback.
    Returns a unified daily DataFrame with forward-filled monthly indicators.
    """
    series_dict = {}
    use_api = False
    
    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            use_api = True
        except Exception as e:
            logging.warning(f"Failed to initialize Fred client with API key: {e}. Falling back to CSV endpoint.")
            use_api = False

    for s_id, s_name in FRED_SERIES_MAP.items():
        s_data = None
        if use_api:
            try:
                s_data = fred.get_series(s_id, observation_start=start_date, observation_end=end_date)
            except Exception as e:
                logging.warning(f"FRED API call failed for {s_id}: {e}. Trying fallback URL...")
                s_data = fetch_fred_csv_fallback(s_id)
        else:
            s_data = fetch_fred_csv_fallback(s_id)
            
        if s_data is not None and not s_data.empty:
            s_data = s_data.rename(s_name)
            series_dict[s_name] = s_data
            
    if not series_dict:
        raise ValueError("Unable to fetch any macroeconomic series from FRED.")
        
    macro_df = pd.DataFrame(series_dict)
    
    # Filter dates
    if start_date:
        macro_df = macro_df[macro_df.index >= pd.to_datetime(start_date)]
    if end_date:
        macro_df = macro_df[macro_df.index <= pd.to_datetime(end_date)]
        
    macro_df = macro_df.sort_index()
    # Forward fill lower frequency macro data (like CPI/UNRATE) across trading days
    macro_df = macro_df.ffill()
    return macro_df

def fetch_etf_prices(tickers: list = None, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Downloads historical Adjusted Close prices for specified ETF tickers + SPY benchmark.
    """
    if tickers is None:
        tickers = ["SPY", "XLE", "XLF", "XLK", "XLU", "XLV", "XLI", "XLY", "XLP", "XLC", "XLB", "XLRE"]
    else:
        if "SPY" not in tickers:
            tickers = list(tickers) + ["SPY"]

    logging.info(f"Downloading price data for {len(tickers)} tickers via yfinance...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.levels[0]:
                prices = data["Adj Close"]
            elif "Close" in data.columns.levels[0]:
                prices = data["Close"]
            else:
                prices = data.iloc[:, :len(tickers)]
        else:
            prices = data

        # Drop any ticker column that has all NaNs
        prices = prices.dropna(how="all", axis=1)
        prices = prices.ffill().bfill()
        return prices
    except Exception as e:
        logging.error(f"Error downloading ETF prices: {e}")
        raise e

def load_combined_dataset(start_date: str = "2010-01-01", end_date: str = None, fred_api_key: str = None):
    """
    Downloads ETF prices and Macro data, aligning them on daily business trading calendar.
    """
    prices = fetch_etf_prices(start_date=start_date, end_date=end_date)
    macro = fetch_macro_data(api_key=fred_api_key, start_date=start_date, end_date=end_date)
    
    # Align on common datetime index (inner join or left join on price trading calendar)
    combined = prices.join(macro, how="inner")
    combined = combined.ffill().bfill()
    return combined, prices, macro
