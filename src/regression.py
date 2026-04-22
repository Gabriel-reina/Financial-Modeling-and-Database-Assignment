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


def load_portfolio_returns():
    return pd.read_parquet(os.path.join(PROCESSED_DIR, "portfolio_returns.parquet"))


def table1_summary_statistics():
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    results = []
    for factor_name in port_ret["factor_name"].unique():
        if factor_name not in FACTOR_PUBLICATION_INFO:
            continue
        factor_data = port_ret[port_ret["factor_name"] == factor_name]
        data_start, is_end, oos_end = _get_periods(factor_name)
        sample_end_year, pub_year = FACTOR_PUBLICATION_INFO[factor_name]

        periods = {
            f"In-Sample (~{sample_end_year})": (data_start, is_end),
            f"Out-of-Sample ({sample_end_year+1}-{pub_year})": (f"{max(sample_end_year,1990)+1}-01-01", oos_end),
            f"Post-Publication (>{pub_year})": (f"{pub_year+1}-01-01", "2099-12-31"),
        }

        row = {
            "Factor": factor_name,
            "Category": FACTOR_CATEGORIES.get(factor_name, "Other"),
            "SampleEndYear": sample_end_year,
            "PubYear": pub_year,
        }

        for period_name, (start, end) in periods.items():
            sub = factor_data[
                (factor_data["month"] >= start) & (factor_data["month"] <= end)
            ]
            if len(sub) > 0:
                mean_ret = sub["ls_return"].mean()
                std_ret = sub["ls_return"].std()
                t_stat = mean_ret / (std_ret / np.sqrt(len(sub))) if std_ret > 0 else 0
                n_months = len(sub)
            else:
                mean_ret = std_ret = t_stat = np.nan
                n_months = 0

            prefix = period_name.split("(")[0].strip()
            row[f"{prefix} Mean(%)"] = mean_ret * 100
            row[f"{prefix} Std(%)"] = std_ret * 100
            row[f"{prefix} t-stat"] = t_stat
            row[f"{prefix} N"] = n_months

        results.append(row)

    df_result = pd.DataFrame(results)

    summary_row = {"Factor": "Average", "Category": "All"}
    for col in df_result.columns:
        if col in ("Factor", "Category"):
            continue
        summary_row[col] = df_result[col].mean()
    df_result = pd.concat([df_result, pd.DataFrame([summary_row])], ignore_index=True)

    output_path = os.path.join(OUTPUT_DIR, "table1_summary_statistics.csv")
    df_result.to_csv(output_path, index=False, float_format="%.4f")
    return df_result


def table2_main_regression():
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    rows = []
    valid_factors = []
    for factor_name in port_ret["factor_name"].unique():
        if factor_name not in FACTOR_PUBLICATION_INFO:
            continue
        sub = port_ret[port_ret["factor_name"] == factor_name].copy()
        data_start, is_end, oos_end = _get_periods(factor_name)

        in_sample = sub[(sub["month"] >= data_start) & (sub["month"] <= is_end)]
        if len(in_sample) < 12:
            continue
        valid_factors.append(factor_name)

        sub["post_sample"] = (
            (sub["month"] > is_end) & (sub["month"] <= oos_end)
        ).astype(int)
        sub["post_publication"] = (sub["month"] > oos_end).astype(int)
        rows.append(sub)

    port_ret = pd.concat(rows, ignore_index=True)
    port_ret["ls_return_pct"] = port_ret["ls_return"] * 100

    from numpy.linalg import lstsq

    dependent = port_ret["ls_return_pct"].values
    intercept = np.ones(len(port_ret))
    factor_dummies = pd.get_dummies(port_ret["factor_name"], drop_first=True, dtype=float)

    explanatory_vars = np.column_stack([
        intercept,
        factor_dummies.values,
        port_ret["post_sample"].values,
        port_ret["post_publication"].values,
    ])

    beta, _, _, _ = lstsq(explanatory_vars, dependent, rcond=None)
    resid = dependent - explanatory_vars @ beta

    n_factors = len(valid_factors)
    beta_post_sample = beta[-2]
    beta_post_pub = beta[-1]

    se_post_sample, se_post_pub = _clustered_se(
        port_ret, explanatory_vars, resid, [-2, -1], cluster_col="month"
    )

    t_post_sample = beta_post_sample / se_post_sample
    t_post_pub = beta_post_pub / se_post_pub

    is_masks = []
    for factor_name in valid_factors:
        data_start, is_end, _ = _get_periods(factor_name)
        mask = (
            (port_ret["factor_name"] == factor_name)
            & (port_ret["month"] >= data_start)
            & (port_ret["month"] <= is_end)
        )
        is_masks.append(mask)
    in_sample_mask = pd.concat([pd.Series(m) for m in is_masks], axis=1).any(axis=1)
    in_sample_mean = port_ret.loc[in_sample_mask, "ls_return_pct"].mean()

    decay_oos = beta_post_sample / in_sample_mean * 100 if in_sample_mean != 0 else np.nan
    decay_post_pub = beta_post_pub / in_sample_mean * 100 if in_sample_mean != 0 else np.nan

    is_mean_by_factor = {}
    for factor_name in valid_factors:
        data_start, is_end, _ = _get_periods(factor_name)
        sub = port_ret[
            (port_ret["factor_name"] == factor_name)
            & (port_ret["month"] >= data_start)
            & (port_ret["month"] <= is_end)
        ]
        is_mean_by_factor[factor_name] = sub["ls_return_pct"].mean() if len(sub) > 0 else 0

    port_ret["in_sample_mean"] = port_ret["factor_name"].map(is_mean_by_factor)
    port_ret["post_pub_x_mean"] = port_ret["post_publication"] * port_ret["in_sample_mean"]
    port_ret["post_sample_x_mean"] = port_ret["post_sample"] * port_ret["in_sample_mean"]

    explanatory_vars2 = np.column_stack([
        intercept,
        factor_dummies.values,
        port_ret["post_sample"].values,
        port_ret["post_publication"].values,
        port_ret["post_sample_x_mean"].values,
        port_ret["post_pub_x_mean"].values,
    ])

    beta2, _, _, _ = lstsq(explanatory_vars2, dependent, rcond=None)
    resid2 = dependent - explanatory_vars2 @ beta2

    se2 = _clustered_se(
        port_ret, explanatory_vars2, resid2,
        [-4, -3, -2, -1], cluster_col="month"
    )

    result_dict = {
        "in_sample_mean_pct": in_sample_mean,
        "beta_post_sample": beta_post_sample,
        "se_post_sample": se_post_sample,
        "t_post_sample": t_post_sample,
        "beta_post_pub": beta_post_pub,
        "se_post_pub": se_post_pub,
        "t_post_pub": t_post_pub,
        "decay_oos_pct": decay_oos,
        "decay_post_pub_pct": decay_post_pub,
        "n_obs": len(port_ret),
        "n_factors": n_factors,
    }
    pd.DataFrame([result_dict]).to_csv(
        os.path.join(OUTPUT_DIR, "table2_main_regression.csv"), index=False
    )
    return result_dict


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


def table3_cross_sectional_decay():
    """
    Table III: Cross-Sectional Decay by Factor
    Shows the decay pattern from In-Sample → Out-of-Sample → Post-Publication
    """
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    factor_stats = []
    for factor_name in port_ret["factor_name"].unique():
        if factor_name not in FACTOR_PUBLICATION_INFO:
            continue
        sub = port_ret[port_ret["factor_name"] == factor_name]
        data_start, is_end, oos_end = _get_periods(factor_name)
        sample_end_year, pub_year = FACTOR_PUBLICATION_INFO[factor_name]

        in_sample = sub[(sub["month"] >= data_start) & (sub["month"] <= is_end)]
        out_sample = sub[(sub["month"] > is_end) & (sub["month"] <= oos_end)]
        post_pub = sub[sub["month"] > oos_end]

        # Calculate statistics for each period
        is_mean = in_sample["ls_return"].mean() * 100 if len(in_sample) > 0 else np.nan
        is_std = in_sample["ls_return"].std() * 100 if len(in_sample) > 0 else np.nan
        is_t = is_mean / (is_std / np.sqrt(len(in_sample))) if is_std > 0 and len(in_sample) > 0 else np.nan
        is_n = len(in_sample)

        oos_mean = out_sample["ls_return"].mean() * 100 if len(out_sample) > 0 else np.nan
        oos_std = out_sample["ls_return"].std() * 100 if len(out_sample) > 0 else np.nan
        oos_t = oos_mean / (oos_std / np.sqrt(len(out_sample))) if oos_std > 0 and len(out_sample) > 0 else np.nan
        oos_n = len(out_sample)

        pp_mean = post_pub["ls_return"].mean() * 100 if len(post_pub) > 0 else np.nan
        pp_std = post_pub["ls_return"].std() * 100 if len(post_pub) > 0 else np.nan
        pp_t = pp_mean / (pp_std / np.sqrt(len(post_pub))) if pp_std > 0 and len(post_pub) > 0 else np.nan
        pp_n = len(post_pub)

        # Calculate decay from IS to OOS and IS to PP
        decay_is_to_oos = oos_mean - is_mean if not np.isnan(is_mean) and not np.isnan(oos_mean) else np.nan
        decay_is_to_oos_pct = decay_is_to_oos / is_mean * 100 if not np.isnan(is_mean) and is_mean != 0 else np.nan

        decay_is_to_pp = pp_mean - is_mean if not np.isnan(is_mean) and not np.isnan(pp_mean) else np.nan
        decay_is_to_pp_pct = decay_is_to_pp / is_mean * 100 if not np.isnan(is_mean) and is_mean != 0 else np.nan

        # Calculate decay from OOS to PP
        decay_oos_to_pp = pp_mean - oos_mean if not np.isnan(oos_mean) and not np.isnan(pp_mean) else np.nan
        decay_oos_to_pp_pct = decay_oos_to_pp / oos_mean * 100 if not np.isnan(oos_mean) and oos_mean != 0 else np.nan

        factor_stats.append({
            "factor": factor_name,
            "category": FACTOR_CATEGORIES.get(factor_name, "Other"),
            "sample_end_year": sample_end_year,
            "pub_year": pub_year,
            # In-Sample statistics
            "is_mean": is_mean,
            "is_std": is_std,
            "is_t": is_t,
            "is_n": is_n,
            # Out-of-Sample statistics
            "oos_mean": oos_mean,
            "oos_std": oos_std,
            "oos_t": oos_t,
            "oos_n": oos_n,
            # Post-Publication statistics
            "pp_mean": pp_mean,
            "pp_std": pp_std,
            "pp_t": pp_t,
            "pp_n": pp_n,
            # Decay measures
            "decay_is_to_oos": decay_is_to_oos,
            "decay_is_to_oos_pct": decay_is_to_oos_pct,
            "decay_is_to_pp": decay_is_to_pp,
            "decay_is_to_pp_pct": decay_is_to_pp_pct,
            "decay_oos_to_pp": decay_oos_to_pp,
            "decay_oos_to_pp_pct": decay_oos_to_pp_pct,
        })

    df_stats = pd.DataFrame(factor_stats)

    # Cross-sectional regression: decay vs. in-sample strength
    # Only include factors with valid IS and PP data
    valid_for_regression = df_stats[
        (df_stats["is_n"] >= 12) & 
        (df_stats["pp_n"] >= 12) & 
        (df_stats["is_mean"].notna()) & 
        (df_stats["decay_is_to_pp"].notna())
    ]

    if len(valid_for_regression) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_for_regression["is_mean"], valid_for_regression["decay_is_to_pp"]
        )
        slope_t, intercept_t, r_t, p_t, se_t = stats.linregress(
            valid_for_regression["is_t"], valid_for_regression["decay_is_to_pp"]
        )
    else:
        slope = intercept = r_value = p_value = std_err = np.nan
        slope_t = intercept_t = r_t = p_t = se_t = np.nan

    output_path = os.path.join(OUTPUT_DIR, "table3_cross_sectional_decay.csv")
    df_stats.to_csv(output_path, index=False, float_format="%.4f")
    return df_stats

def table4_by_predictor_type():
    """
    Table IV: Decay by Predictor Type
    Shows the decay pattern across different factor categories
    """
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])
    port_ret["category"] = port_ret["factor_name"].map(FACTOR_CATEGORIES)
    port_ret["ls_return_pct"] = port_ret["ls_return"] * 100

    results = []
    for category in ["Market", "Fundamental", "Valuation", "Event"]:
        cat_data = port_ret[port_ret["category"] == category]

        if len(cat_data) == 0:
            continue

        is_parts = []
        oos_parts = []
        pp_parts = []
        for factor_name in cat_data["factor_name"].unique():
            if factor_name not in FACTOR_PUBLICATION_INFO:
                continue
            fdata = cat_data[cat_data["factor_name"] == factor_name]
            data_start, is_end, oos_end = _get_periods(factor_name)
            is_parts.append(fdata[(fdata["month"] >= data_start) & (fdata["month"] <= is_end)])
            oos_parts.append(fdata[(fdata["month"] > is_end) & (fdata["month"] <= oos_end)])
            pp_parts.append(fdata[fdata["month"] > oos_end])

        cat_is = pd.concat(is_parts) if is_parts else pd.DataFrame()
        cat_oos = pd.concat(oos_parts) if oos_parts else pd.DataFrame()
        cat_pp = pd.concat(pp_parts) if pp_parts else pd.DataFrame()

        is_mean = cat_is["ls_return_pct"].mean() if len(cat_is) > 0 else np.nan
        oos_mean = cat_oos["ls_return_pct"].mean() if len(cat_oos) > 0 else np.nan
        pp_mean = cat_pp["ls_return_pct"].mean() if len(cat_pp) > 0 else np.nan

        decay_is_to_oos = oos_mean - is_mean if not np.isnan(is_mean) and not np.isnan(oos_mean) else np.nan
        decay_is_to_pp = pp_mean - is_mean if not np.isnan(is_mean) and not np.isnan(pp_mean) else np.nan
        decay_oos_to_pp = pp_mean - oos_mean if not np.isnan(oos_mean) and not np.isnan(pp_mean) else np.nan

        n_factors = cat_data["factor_name"].nunique()

        results.append({
            "Category": category,
            "N_Factors": n_factors,
            "IS_Mean(%)": is_mean,
            "IS_N_Months": len(cat_is),
            "OOS_Mean(%)": oos_mean,
            "OOS_N_Months": len(cat_oos),
            "PP_Mean(%)": pp_mean,
            "PP_N_Months": len(cat_pp),
            "Decay_IS_to_OOS(%)": decay_is_to_oos,
            "Decay_IS_to_PP(%)": decay_is_to_pp,
            "Decay_OOS_to_PP(%)": decay_oos_to_pp,
        })

    df_result = pd.DataFrame(results)

    output_path = os.path.join(OUTPUT_DIR, "table4_by_predictor_type.csv")
    df_result.to_csv(output_path, index=False, float_format="%.4f")
    return df_result

def table5_arbitrage_costs():
    """
    Table V: Arbitrage Costs and Decay
    Examines the relationship between arbitrage costs and return decay
    """
    panel = pd.read_parquet(os.path.join(PROCESSED_DIR, "all_factors.parquet"))
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    panel["size_rank"] = panel.groupby("month")["mkt_cap_total"].rank(pct=True)

    factor_arb_costs = []
    for factor_name in port_ret["factor_name"].unique():
        if factor_name not in panel.columns:
            continue

        factor_data = panel[["stkcd", "month", factor_name, "size_rank", "mkt_cap_total"]].dropna(
            subset=[factor_name]
        )

        if factor_name == "seo_indicator":
            extreme = factor_data[factor_data[factor_name] == 1]
        else:
            try:
                factor_data["quintile"] = factor_data.groupby("month")[factor_name].transform(
                    lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5]) if len(x) >= 30 else np.nan
                )
                extreme = factor_data[factor_data["quintile"].isin([1, 5])]
            except Exception:
                continue

        if len(extreme) == 0:
            continue

        avg_size_rank = extreme["size_rank"].mean()
        avg_mkt_cap = extreme["mkt_cap_total"].mean()

        sub = port_ret[port_ret["factor_name"] == factor_name]

        if factor_name not in FACTOR_PUBLICATION_INFO:
            continue
        data_start, is_end, oos_end = _get_periods(factor_name)
        in_sample = sub[(sub["month"] >= data_start) & (sub["month"] <= is_end)]
        out_sample = sub[(sub["month"] > is_end) & (sub["month"] <= oos_end)]
        post_pub = sub[sub["month"] > oos_end]

        # Calculate statistics for each period
        is_mean = in_sample["ls_return"].mean() * 100 if len(in_sample) > 0 else np.nan
        oos_mean = out_sample["ls_return"].mean() * 100 if len(out_sample) > 0 else np.nan
        pp_mean = post_pub["ls_return"].mean() * 100 if len(post_pub) > 0 else np.nan

        # Calculate decay measures
        decay_is_to_oos = oos_mean - is_mean if not np.isnan(is_mean) and not np.isnan(oos_mean) else np.nan
        decay_is_to_pp = pp_mean - is_mean if not np.isnan(is_mean) and not np.isnan(pp_mean) else np.nan
        decay_oos_to_pp = pp_mean - oos_mean if not np.isnan(oos_mean) and not np.isnan(pp_mean) else np.nan

        factor_arb_costs.append({
            "factor": factor_name,
            "avg_size_rank": avg_size_rank,
            "avg_mkt_cap_billion": avg_mkt_cap / 1e6,
            "is_mean": is_mean,
            "is_n": len(in_sample),
            "oos_mean": oos_mean,
            "oos_n": len(out_sample),
            "pp_mean": pp_mean,
            "pp_n": len(post_pub),
            "decay_is_to_oos": decay_is_to_oos,
            "decay_is_to_pp": decay_is_to_pp,
            "decay_oos_to_pp": decay_oos_to_pp,
        })

    df_arb = pd.DataFrame(factor_arb_costs)

    # Cross-sectional regression: decay vs. arbitrage costs (size rank)
    # Only include factors with valid IS and PP data
    valid_for_regression = df_arb[
        (df_arb["is_n"] >= 12) & 
        (df_arb["pp_n"] >= 12) & 
        (df_arb["is_mean"].notna()) & 
        (df_arb["decay_is_to_pp"].notna())
    ]

    if len(valid_for_regression) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_for_regression["avg_size_rank"], valid_for_regression["decay_is_to_pp"]
        )
    output_path = os.path.join(OUTPUT_DIR, "table5_arbitrage_costs.csv")
    df_arb.to_csv(output_path, index=False, float_format="%.4f")
    return df_arb

def run_all():
    table1_summary_statistics()
    table2_main_regression()
    table3_cross_sectional_decay()
    table4_by_predictor_type()
    table5_arbitrage_costs()

if __name__ == "__main__":
    run_all()
