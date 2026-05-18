"""
Innovation: Factor Return Heterogeneity across Market Regimes (Bull vs. Bear)

McLean & Pontiff (2016) partition time into IS/OOS/PP but do not consider
market-state conditioning. China's A-share market features dramatic bull-bear
cycles (e.g., 2007, 2015 bubbles). This extension asks:

1. Do anomaly long-short returns differ between bull and bear markets?
2. Is the post-publication decay asymmetric across regimes?
   (Arbitrageurs may face different constraints in different states.)

We define market regime using trailing 12-month equal-weighted market return:
  - Bull: cumulative 12-month market return > 0
  - Bear: cumulative 12-month market return <= 0

Key insight: If post-publication decay is driven by arbitrageurs learning from
papers and trading against mispricing, the decay should be STRONGER in bull
markets (more capital available, lower funding constraints, easier to short via
margin) and WEAKER in bear markets (capital withdrawal, higher constraints).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from portfolio import FACTOR_CATEGORIES, FACTOR_PUBLICATION_INFO

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_START = "1991-01-01"


def _get_periods(factor_name):
    sample_end_year, pub_year = FACTOR_PUBLICATION_INFO[factor_name]
    in_sample_end = f"{max(sample_end_year, 1990)}-12-31"
    out_sample_end = f"{pub_year}-12-31"
    return DATA_START, in_sample_end, out_sample_end


def compute_market_regime():
    """
    Compute monthly market regime (bull=1, bear=0) based on trailing
    12-month equal-weighted market return.
    """
    panel = pd.read_parquet(os.path.join(PROCESSED_DIR, "all_factors.parquet"))
    panel["month"] = pd.to_datetime(panel["month"])

    mkt_ret = panel.groupby("month")["ret"].mean().reset_index()
    mkt_ret.columns = ["month", "mkt_return"]
    mkt_ret = mkt_ret.sort_values("month").reset_index(drop=True)

    mkt_ret["cum_12m"] = mkt_ret["mkt_return"].rolling(12, min_periods=12).sum()
    mkt_ret["bull"] = (mkt_ret["cum_12m"] > 0).astype(int)
    mkt_ret = mkt_ret.dropna(subset=["cum_12m"])

    return mkt_ret[["month", "mkt_return", "cum_12m", "bull"]]


def table6_regime_factor_returns():
    """
    Table VI: Factor Returns Conditional on Market Regime

    For each factor, compute average long-short return separately in
    bull and bear markets, and test whether the difference is significant.
    """
    port_ret = pd.read_parquet(os.path.join(PROCESSED_DIR, "portfolio_returns.parquet"))
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    regime = compute_market_regime()
    port_ret = port_ret.merge(regime[["month", "bull"]], on="month", how="inner")

    results = []
    for factor_name in sorted(port_ret["factor_name"].unique()):
        sub = port_ret[port_ret["factor_name"] == factor_name]

        bull_data = sub[sub["bull"] == 1]["ls_return"]
        bear_data = sub[sub["bull"] == 0]["ls_return"]

        bull_mean = bull_data.mean() * 100
        bull_std = bull_data.std() * 100
        bull_t = bull_mean / (bull_std / np.sqrt(len(bull_data))) if bull_std > 0 else np.nan
        bull_n = len(bull_data)

        bear_mean = bear_data.mean() * 100
        bear_std = bear_data.std() * 100
        bear_t = bear_mean / (bear_std / np.sqrt(len(bear_data))) if bear_std > 0 else np.nan
        bear_n = len(bear_data)

        diff_mean = bull_mean - bear_mean
        pooled_se = np.sqrt(bull_std**2 / bull_n + bear_std**2 / bear_n) if bull_n > 0 and bear_n > 0 else np.nan
        diff_t = diff_mean / pooled_se if pooled_se and pooled_se > 0 else np.nan

        results.append({
            "factor": factor_name,
            "category": FACTOR_CATEGORIES.get(factor_name, "Other"),
            "bull_mean_pct": bull_mean,
            "bull_t": bull_t,
            "bull_n": bull_n,
            "bear_mean_pct": bear_mean,
            "bear_t": bear_t,
            "bear_n": bear_n,
            "diff_bull_minus_bear": diff_mean,
            "diff_t": diff_t,
        })

    df_result = pd.DataFrame(results)

    avg_row = {
        "factor": "Average",
        "category": "All",
        "bull_mean_pct": df_result["bull_mean_pct"].mean(),
        "bull_t": np.nan,
        "bull_n": np.nan,
        "bear_mean_pct": df_result["bear_mean_pct"].mean(),
        "bear_t": np.nan,
        "bear_n": np.nan,
        "diff_bull_minus_bear": df_result["diff_bull_minus_bear"].mean(),
        "diff_t": np.nan,
    }
    df_result = pd.concat([df_result, pd.DataFrame([avg_row])], ignore_index=True)

    output_path = os.path.join(OUTPUT_DIR, "table6_regime_factor_returns.csv")
    df_result.to_csv(output_path, index=False, float_format="%.4f")
    print("Table VI: Factor Returns by Market Regime")
    print("=" * 80)
    print(df_result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    return df_result


def table7_regime_decay_interaction():
    """
    Table VII: Post-Publication Decay × Market Regime Interaction

    Regression:
      R_{i,t} = alpha_i + beta1 * PostPub_{i,t} + beta2 * Bear_t
              + beta3 * PostPub_{i,t} * Bear_t + epsilon_{i,t}

    beta3 tests whether post-publication decay differs between bull and bear.
    If arbitrage is easier in bull markets, beta3 should be positive
    (less decay in bear markets).

    We run this on ALL factors (using all available data), treating everything
    before the publication date as "pre-publication" for each factor.
    """
    port_ret = pd.read_parquet(os.path.join(PROCESSED_DIR, "portfolio_returns.parquet"))
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    regime = compute_market_regime()
    port_ret = port_ret.merge(regime[["month", "bull"]], on="month", how="inner")
    port_ret["bear"] = 1 - port_ret["bull"]

    rows = []
    for factor_name in port_ret["factor_name"].unique():
        if factor_name not in FACTOR_PUBLICATION_INFO:
            continue
        sub = port_ret[port_ret["factor_name"] == factor_name].copy()
        _, _, oos_end = _get_periods(factor_name)
        sub["post_pub"] = (sub["month"] > oos_end).astype(int)
        rows.append(sub)

    port_ret = pd.concat(rows, ignore_index=True)
    port_ret["ls_return_pct"] = port_ret["ls_return"] * 100
    port_ret["post_pub_x_bear"] = port_ret["post_pub"] * port_ret["bear"]

    from numpy.linalg import lstsq

    intercept = np.ones(len(port_ret))
    factor_dummies = pd.get_dummies(port_ret["factor_name"], drop_first=True, dtype=float)

    explanatory_vars = np.column_stack([
        intercept,
        factor_dummies.values,
        port_ret["post_pub"].values,
        port_ret["bear"].values,
        port_ret["post_pub_x_bear"].values,
    ])

    dependent = port_ret["ls_return_pct"].values
    beta, _, _, _ = lstsq(explanatory_vars, dependent, rcond=None)
    resid = dependent - explanatory_vars @ beta

    beta_post_pub = beta[-3]
    beta_bear = beta[-2]
    beta_interaction = beta[-1]

    se_list = _clustered_se(port_ret, explanatory_vars, resid, [-3, -2, -1], cluster_col="month")
    se_post_pub, se_bear, se_interaction = se_list

    t_post_pub = beta_post_pub / se_post_pub
    t_bear = beta_bear / se_bear
    t_interaction = beta_interaction / se_interaction

    result = {
        "beta_post_pub": beta_post_pub,
        "se_post_pub": se_post_pub,
        "t_post_pub": t_post_pub,
        "beta_bear": beta_bear,
        "se_bear": se_bear,
        "t_bear": t_bear,
        "beta_post_pub_x_bear": beta_interaction,
        "se_interaction": se_interaction,
        "t_interaction": t_interaction,
        "n_obs": len(port_ret),
        "n_factors": port_ret["factor_name"].nunique(),
    }

    pd.DataFrame([result]).to_csv(
        os.path.join(OUTPUT_DIR, "table7_regime_decay_interaction.csv"),
        index=False, float_format="%.4f"
    )

    print("Table VII: Post-Publication Decay × Market Regime Interaction")
    print("=" * 80)
    print(f"  R_it = alpha_i + beta1*PostPub + beta2*Bear + beta3*PostPub×Bear + eps")
    print()
    print(f"  {'Variable':<25} {'Coefficient':>12} {'Clustered SE':>14} {'t-stat':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*10}")
    print(f"  {'Post-Publication':<25} {beta_post_pub:>12.4f} {se_post_pub:>14.4f} {t_post_pub:>10.3f}")
    print(f"  {'Bear Market':<25} {beta_bear:>12.4f} {se_bear:>14.4f} {t_bear:>10.3f}")
    print(f"  {'PostPub × Bear':<25} {beta_interaction:>12.4f} {se_interaction:>14.4f} {t_interaction:>10.3f}")
    print(f"\n  Observations: {len(port_ret)}, Factors: {port_ret['factor_name'].nunique()}")
    print()
    return result


def table8_regime_by_factor():
    """
    Table VIII: Factor-Level Returns in Bull vs Bear, Pre- vs Post-Publication

    For each factor, show the 2x2: {Pre-Pub, Post-Pub} × {Bull, Bear}
    """
    port_ret = pd.read_parquet(os.path.join(PROCESSED_DIR, "portfolio_returns.parquet"))
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    regime = compute_market_regime()
    port_ret = port_ret.merge(regime[["month", "bull"]], on="month", how="inner")

    results = []
    for factor_name in sorted(port_ret["factor_name"].unique()):
        if factor_name not in FACTOR_PUBLICATION_INFO:
            continue
        sub = port_ret[port_ret["factor_name"] == factor_name]
        _, _, oos_end = _get_periods(factor_name)

        pre_pub = sub[sub["month"] <= oos_end]
        post_pub = sub[sub["month"] > oos_end]

        pre_bull = pre_pub[pre_pub["bull"] == 1]["ls_return"]
        pre_bear = pre_pub[pre_pub["bull"] == 0]["ls_return"]
        post_bull = post_pub[post_pub["bull"] == 1]["ls_return"]
        post_bear = post_pub[post_pub["bull"] == 0]["ls_return"]

        row = {
            "factor": factor_name,
            "category": FACTOR_CATEGORIES.get(factor_name, "Other"),
            "pre_pub_bull_mean": pre_bull.mean() * 100 if len(pre_bull) > 0 else np.nan,
            "pre_pub_bull_n": len(pre_bull),
            "pre_pub_bear_mean": pre_bear.mean() * 100 if len(pre_bear) > 0 else np.nan,
            "pre_pub_bear_n": len(pre_bear),
            "post_pub_bull_mean": post_bull.mean() * 100 if len(post_bull) > 0 else np.nan,
            "post_pub_bull_n": len(post_bull),
            "post_pub_bear_mean": post_bear.mean() * 100 if len(post_bear) > 0 else np.nan,
            "post_pub_bear_n": len(post_bear),
        }

        # Decay in bull market
        if not np.isnan(row["pre_pub_bull_mean"]) and not np.isnan(row["post_pub_bull_mean"]) and row["pre_pub_bull_n"] >= 6:
            row["decay_bull"] = row["post_pub_bull_mean"] - row["pre_pub_bull_mean"]
        else:
            row["decay_bull"] = np.nan

        # Decay in bear market
        if not np.isnan(row["pre_pub_bear_mean"]) and not np.isnan(row["post_pub_bear_mean"]) and row["pre_pub_bear_n"] >= 6:
            row["decay_bear"] = row["post_pub_bear_mean"] - row["pre_pub_bear_mean"]
        else:
            row["decay_bear"] = np.nan

        results.append(row)

    df_result = pd.DataFrame(results)

    output_path = os.path.join(OUTPUT_DIR, "table8_regime_by_factor.csv")
    df_result.to_csv(output_path, index=False, float_format="%.4f")

    print("Table VIII: Factor Returns in Bull vs Bear, Pre- vs Post-Publication")
    print("=" * 80)
    print(df_result[["factor", "pre_pub_bull_mean", "pre_pub_bear_mean",
                     "post_pub_bull_mean", "post_pub_bear_mean",
                     "decay_bull", "decay_bear"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    return df_result


def _clustered_se(df, X, residuals, coef_indices, cluster_col="month"):
    clusters = df[cluster_col].values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    n_obs = len(residuals)
    n_params = X.shape[1]

    XtX_inv = np.linalg.pinv(X.T @ X)

    meat = np.zeros((n_params, n_params))
    for cluster_val in unique_clusters:
        mask = clusters == cluster_val
        X_c = X[mask]
        e_c = residuals[mask]
        score_c = X_c.T @ e_c
        meat += np.outer(score_c, score_c)

    correction = n_clusters / (n_clusters - 1) * (n_obs - 1) / (n_obs - n_params)
    sandwich = correction * XtX_inv @ meat @ XtX_inv

    standard_errors = []
    for idx in coef_indices:
        se = np.sqrt(sandwich[idx, idx])
        standard_errors.append(se)

    return standard_errors


def run_innovation():
    print("\n" + "=" * 80)
    print("INNOVATION: Factor Return Heterogeneity across Market Regimes")
    print("=" * 80 + "\n")

    table6_regime_factor_returns()
    table7_regime_decay_interaction()
    table8_regime_by_factor()

    print("All innovation results saved to output/")


if __name__ == "__main__":
    run_innovation()
