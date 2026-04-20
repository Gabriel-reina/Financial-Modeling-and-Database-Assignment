import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IN_SAMPLE_START = "2000-01-01"
IN_SAMPLE_END = "2010-12-31"
OUT_SAMPLE_END = "2015-12-31"
POST_PUB_END = "2026-12-31"

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


def load_portfolio_returns():
    return pd.read_parquet(os.path.join(PROCESSED_DIR, "portfolio_returns.parquet"))


def table1_summary_statistics():
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    periods = {
        "In-Sample (2000-2010)": (IN_SAMPLE_START, IN_SAMPLE_END),
        "Out-of-Sample (2011-2015)": ("2011-01-01", OUT_SAMPLE_END),
        "Post-Publication (2016-2026)": ("2016-01-01", POST_PUB_END),
    }

    results = []
    for factor_name in port_ret["factor_name"].unique():
        factor_data = port_ret[port_ret["factor_name"] == factor_name]
        row = {"Factor": factor_name, "Category": FACTOR_CATEGORIES.get(factor_name, "Other")}

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

    # 保存
    output_path = os.path.join(OUTPUT_DIR, "table1_summary_statistics.csv")
    df_result.to_csv(output_path, index=False, float_format="%.4f")
    return df_result


def table2_main_regression():
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    valid_factors = []
    for factor_name in port_ret["factor_name"].unique():
        sub = port_ret[port_ret["factor_name"] == factor_name]
        in_sample = sub[(sub["month"] >= IN_SAMPLE_START) & (sub["month"] <= IN_SAMPLE_END)]
        if len(in_sample) >= 24:
            valid_factors.append(factor_name)

    port_ret = port_ret[port_ret["factor_name"].isin(valid_factors)].copy()

    port_ret["post_sample"] = (
        (port_ret["month"] > IN_SAMPLE_END) & (port_ret["month"] <= OUT_SAMPLE_END)
    ).astype(int)
    port_ret["post_publication"] = (port_ret["month"] > OUT_SAMPLE_END).astype(int)

    port_ret["ls_return_pct"] = port_ret["ls_return"] * 100

    factor_means = port_ret.groupby("factor_name")["ls_return_pct"].transform("mean")
    port_ret["ls_return_demean"] = port_ret["ls_return_pct"] - factor_means

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

    beta, residuals, rank, sv = lstsq(explanatory_vars, dependent, rcond=None)
    fitted = explanatory_vars @ beta
    resid = dependent - fitted

    n_factors = len(valid_factors)
    beta_post_sample = beta[-2]
    beta_post_pub = beta[-1]

    se_post_sample, se_post_pub = _clustered_se(
        port_ret, explanatory_vars, resid, [-2, -1], cluster_col="month"
    )

    t_post_sample = beta_post_sample / se_post_sample
    t_post_pub = beta_post_pub / se_post_pub

    in_sample_data = port_ret[
        (port_ret["month"] >= IN_SAMPLE_START) & (port_ret["month"] <= IN_SAMPLE_END)
    ]
    in_sample_mean = in_sample_data["ls_return_pct"].mean()

    decay_oos = beta_post_sample / in_sample_mean * 100 if in_sample_mean != 0 else np.nan
    decay_post_pub = beta_post_pub / in_sample_mean * 100 if in_sample_mean != 0 else np.nan

    port_ret["in_sample_mean"] = port_ret["factor_name"].map(
        in_sample_data.groupby("factor_name")["ls_return_pct"].mean()
    )
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
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])

    factor_stats = []
    for factor_name in port_ret["factor_name"].unique():
        sub = port_ret[port_ret["factor_name"] == factor_name]

        in_sample = sub[(sub["month"] >= IN_SAMPLE_START) & (sub["month"] <= IN_SAMPLE_END)]
        post_pub = sub[sub["month"] > OUT_SAMPLE_END]

        if len(in_sample) < 12 or len(post_pub) < 12:
            continue

        is_mean = in_sample["ls_return"].mean() * 100
        is_std = in_sample["ls_return"].std() * 100
        is_t = is_mean / (is_std / np.sqrt(len(in_sample))) if is_std > 0 else 0
        pp_mean = post_pub["ls_return"].mean() * 100

        decay = pp_mean - is_mean
        decay_pct = decay / is_mean * 100 if is_mean != 0 else np.nan

        factor_stats.append({
            "factor": factor_name,
            "category": FACTOR_CATEGORIES.get(factor_name, "Other"),
            "in_sample_mean": is_mean,
            "in_sample_t": is_t,
            "post_pub_mean": pp_mean,
            "decay": decay,
            "decay_pct": decay_pct,
        })

    df_stats = pd.DataFrame(factor_stats)

    if len(df_stats) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_stats["in_sample_mean"], df_stats["decay"]
        )
        slope_t, intercept_t, r_t, p_t, se_t = stats.linregress(
            df_stats["in_sample_t"], df_stats["decay"]
        )

    output_path = os.path.join(OUTPUT_DIR, "table3_cross_sectional_decay.csv")
    df_stats.to_csv(output_path, index=False, float_format="%.4f")
    return df_stats


def table4_by_predictor_type():
    port_ret = load_portfolio_returns()
    port_ret["month"] = pd.to_datetime(port_ret["month"])
    port_ret["category"] = port_ret["factor_name"].map(FACTOR_CATEGORIES)
    port_ret["post_publication"] = (port_ret["month"] > OUT_SAMPLE_END).astype(int)
    port_ret["ls_return_pct"] = port_ret["ls_return"] * 100

    results = []
    for category in ["Market", "Fundamental", "Valuation", "Event"]:
        cat_data = port_ret[port_ret["category"] == category]
        other_data = port_ret[port_ret["category"] != category]

        if len(cat_data) == 0:
            continue

        cat_is = cat_data[
            (cat_data["month"] >= IN_SAMPLE_START) & (cat_data["month"] <= IN_SAMPLE_END)
        ]
        cat_pp = cat_data[cat_data["month"] > OUT_SAMPLE_END]

        is_mean = cat_is["ls_return_pct"].mean() if len(cat_is) > 0 else np.nan
        pp_mean = cat_pp["ls_return_pct"].mean() if len(cat_pp) > 0 else np.nan
        decay = pp_mean - is_mean

        n_factors = cat_data["factor_name"].nunique()

        results.append({
            "Category": category,
            "N_Factors": n_factors,
            "IS_Mean(%)": is_mean,
            "PP_Mean(%)": pp_mean,
            "Decay(%)": decay,
            "N_months_IS": len(cat_is),
            "N_months_PP": len(cat_pp),
        })

    df_result = pd.DataFrame(results)

    output_path = os.path.join(OUTPUT_DIR, "table4_by_predictor_type.csv")
    df_result.to_csv(output_path, index=False, float_format="%.4f")
    return df_result


def table5_arbitrage_costs():
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
        in_sample = sub[(sub["month"] >= IN_SAMPLE_START) & (sub["month"] <= IN_SAMPLE_END)]
        post_pub = sub[sub["month"] > OUT_SAMPLE_END]

        if len(in_sample) < 12 or len(post_pub) < 12:
            continue

        is_mean = in_sample["ls_return"].mean() * 100
        pp_mean = post_pub["ls_return"].mean() * 100
        decay = pp_mean - is_mean

        factor_arb_costs.append({
            "factor": factor_name,
            "avg_size_rank": avg_size_rank,
            "avg_mkt_cap_billion": avg_mkt_cap / 1e6,
            "is_mean": is_mean,
            "pp_mean": pp_mean,
            "decay": decay,
        })

    df_arb = pd.DataFrame(factor_arb_costs)

    if len(df_arb) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_arb["avg_size_rank"], df_arb["decay"]
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
