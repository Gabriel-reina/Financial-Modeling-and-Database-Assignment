import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FACTOR_DIRECTIONS = {
    "size": "low",
    "momentum": "high",
    "reversal": "low",
    "turnover": "low",
    "idiovol": "low",
    "leverage": "low",
    "asset_growth": "low",
    "accruals_ratio": "low",
    "roa": "high",
    "bm": "high",
    "analyst_rating_chg": "high",
    "seo_indicator": "low",
}

FACTOR_CATEGORIES = {
    "size": "Market",
    "momentum": "Market",
    "reversal": "Market",
    "turnover": "Market",
    "idiovol": "Market",
    "leverage": "Fundamental",
    "asset_growth": "Fundamental",
    "accruals_ratio": "Fundamental",
    "roa": "Fundamental",
    "bm": "Valuation",
    "analyst_rating_chg": "Event",
    "seo_indicator": "Event",
}

# (original sample end year, publication year) from McLean & Pontiff (2016) Table IA.III
FACTOR_PUBLICATION_INFO = {
    "size":               (1975, 1981),   # Banz (JFE 1981), sample 1926-1975
    "momentum":           (1989, 1993),   # Jegadeesh & Titman (JF 1993), sample 1964-1989
    "reversal":           (1987, 1990),   # Jegadeesh (1989, published 1990), sample 1934-1987
    "turnover":           (1991, 1998),   # Datar, Naik & Radcliffe (JFM 1998), sample 1962-1991
    "idiovol":            (2000, 2006),   # Ang et al. (JF 2006), sample 1986-2000
    "leverage":           (1981, 1988),   # Bhandari (JFE 1988), sample 1946-1981
    "asset_growth":       (2003, 2008),   # Cooper, Gulen & Schill (JF 2008), sample 1968-2003
    "accruals_ratio":     (1991, 1996),   # Sloan (AR 1996), sample 1962-1991
    "roa":                (1993, 1996),   # Haugen & Baker (JFE 1996), sample 1979-1993
    "bm":                 (1990, 1992),   # Fama & French (JF 1992), sample 1963-1990
    "analyst_rating_chg": (1998, 2004),   # Jegadeesh et al. (JF 2004), sample 1985-1998
    "seo_indicator":      (1984, 1995),   # Loughran & Ritter (JF 1995), sample 1975-1984
}


def build_long_short_portfolios(panel=None, min_stocks_per_group=30):
    output_path = os.path.join(PROCESSED_DIR, "portfolio_returns.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)

    if panel is None:
        panel = pd.read_parquet(os.path.join(PROCESSED_DIR, "all_factors.parquet"))

    all_results = []

    for factor_name, direction in FACTOR_DIRECTIONS.items():
        if factor_name not in panel.columns:
            continue
        df = panel[["stkcd", "month", "ret", factor_name]].dropna().copy()

        if factor_name == "seo_indicator":
            results = _build_indicator_portfolio(df, factor_name, direction)
        else:
            results = _build_quintile_portfolio(df, factor_name, direction, min_stocks_per_group)

        all_results.append(results)

    portfolio_returns = pd.concat(all_results, ignore_index=True)
    portfolio_returns.sort_values(["factor_name", "month"], inplace=True)
    portfolio_returns.reset_index(drop=True, inplace=True)

    portfolio_returns.to_parquet(output_path, index=False)
    return portfolio_returns


def _build_quintile_portfolio(df, factor_name, direction, min_stocks=30):
    results = []

    for month, group in df.groupby("month"):
        if len(group) < min_stocks * 2:
            continue

        try:
            group["quintile"] = pd.qcut(group[factor_name], 5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            continue

        if direction == "high":
            long_group = group[group["quintile"] == 5]
            short_group = group[group["quintile"] == 1]
        else:
            long_group = group[group["quintile"] == 1]
            short_group = group[group["quintile"] == 5]

        if len(long_group) < 5 or len(short_group) < 5:
            continue

        long_ret = long_group["ret"].mean()
        short_ret = short_group["ret"].mean()

        results.append({
            "month": month,
            "factor_name": factor_name,
            "ls_return": long_ret - short_ret,
            "long_return": long_ret,
            "short_return": short_ret,
            "n_long": len(long_group),
            "n_short": len(short_group),
        })

    return pd.DataFrame(results)


def _build_indicator_portfolio(df, factor_name, direction):
    results = []

    for month, group in df.groupby("month"):
        indicated = group[group[factor_name] == 1]
        non_indicated = group[group[factor_name] == 0]

        if len(indicated) < 5 or len(non_indicated) < 5:
            continue

        indicated_ret = indicated["ret"].mean()
        non_indicated_ret = non_indicated["ret"].mean()

        if direction == "low":
            long_ret = non_indicated_ret
            short_ret = indicated_ret
            n_long = len(non_indicated)
            n_short = len(indicated)
        else:
            long_ret = indicated_ret
            short_ret = non_indicated_ret
            n_long = len(indicated)
            n_short = len(non_indicated)

        results.append({
            "month": month,
            "factor_name": factor_name,
            "ls_return": long_ret - short_ret,
            "long_return": long_ret,
            "short_return": short_ret,
            "n_long": n_long,
            "n_short": n_short,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import time
    start = time.time()
    portfolio_returns = build_long_short_portfolios()
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s")