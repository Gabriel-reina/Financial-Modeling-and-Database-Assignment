import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _load(name):
    return pd.read_parquet(os.path.join(PROCESSED_DIR, f"{name}.parquet"))


def build_stock_universe():
    output_path = os.path.join(PROCESSED_DIR, "stock_universe.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)
    monthly = _load("monthly_return")

    # 仅保留 A 股
    a_share_types = [1, 4, 16, 32]
    monthly = monthly[monthly["market_type"].isin(a_share_types)].copy()

    # 剔除上市不满 6 个月的新股
    monthly["months_since_ipo"] = (
        (monthly["month"].dt.year - monthly["list_date"].dt.year) * 12
        + (monthly["month"].dt.month - monthly["list_date"].dt.month)
    )
    monthly = monthly[monthly["months_since_ipo"] >= 6].copy()

    # 剔除收益率缺失
    monthly = monthly.dropna(subset=["ret"]).copy()

    monthly.sort_values(["stkcd", "month"], inplace=True)
    monthly.reset_index(drop=True, inplace=True)

    monthly.to_parquet(output_path, index=False)
    return monthly


def factor_size(universe):
    df = universe[["stkcd", "month", "mkt_cap_total"]].copy()
    df["size"] = np.log(df["mkt_cap_total"].replace(0, np.nan))
    return df[["stkcd", "month", "size"]].dropna()


def factor_momentum(universe):
    df = universe[["stkcd", "month", "ret"]].copy()
    df.sort_values(["stkcd", "month"], inplace=True)

    df["log_ret"] = np.log1p(df["ret"])
    grouped = df.groupby("stkcd")["log_ret"]

    df["cum_12"] = grouped.transform(lambda x: x.rolling(12, min_periods=12).sum())
    df["cum_1"] = df["log_ret"]
    # 动量 = 过去 12 个月 - 最近 1 个月 (即 t-12 到 t-2)
    df["momentum"] = df["cum_12"] - df["cum_1"]

    return df[["stkcd", "month", "momentum"]].dropna()


def factor_reversal(universe):
    df = universe[["stkcd", "month", "ret"]].copy()
    df.sort_values(["stkcd", "month"], inplace=True)
    df["reversal"] = df.groupby("stkcd")["ret"].shift(1)
    return df[["stkcd", "month", "reversal"]].dropna()


def factor_turnover(universe):
    daily = _load("daily_return")
    daily = daily[["stkcd", "date", "volume", "mkt_cap_float"]].copy()
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    daily["daily_turnover"] = daily["volume"] / daily["mkt_cap_float"].replace(0, np.nan)

    monthly_turnover = (
        daily.groupby(["stkcd", "month"])["daily_turnover"]
        .mean()
        .reset_index()
        .rename(columns={"daily_turnover": "avg_turnover"})
    )

    monthly_turnover.sort_values(["stkcd", "month"], inplace=True)
    monthly_turnover["turnover"] = (
        monthly_turnover.groupby("stkcd")["avg_turnover"]
        .transform(lambda x: x.rolling(3, min_periods=2).mean())
    )

    result = universe[["stkcd", "month"]].merge(
        monthly_turnover[["stkcd", "month", "turnover"]], on=["stkcd", "month"], how="left"
    )
    return result[["stkcd", "month", "turnover"]].dropna()


def factor_idiovol(universe):
    daily = _load("daily_return")
    daily = daily[["stkcd", "date", "ret"]].dropna().copy()
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    market_ret = daily.groupby("date")["ret"].mean().reset_index()
    market_ret.rename(columns={"ret": "mkt_ret"}, inplace=True)
    daily = daily.merge(market_ret, on="date", how="left")

    def calc_idiovol(group):
        if len(group) < 10:
            return np.nan
        excess = group["ret"].values - group["mkt_ret"].values
        return np.std(excess, ddof=1)

    idiovol = (
        daily.groupby(["stkcd", "month"])
        .apply(calc_idiovol, include_groups=False)
        .reset_index()
        .rename(columns={0: "idiovol"})
    )

    result = universe[["stkcd", "month"]].merge(
        idiovol, on=["stkcd", "month"], how="left"
    )
    return result[["stkcd", "month", "idiovol"]].dropna()


def _prepare_annual_fundamentals():
    bs = _load("balance_sheet")
    bs = bs[bs["report_date"].dt.month == 12].copy()
    bs["year"] = bs["report_date"].dt.year
    bs.sort_values(["stkcd", "year"], inplace=True)
    bs.drop_duplicates(subset=["stkcd", "year"], keep="last", inplace=True)
    return bs


def factor_leverage(universe):
    bs = _prepare_annual_fundamentals()
    bs["leverage"] = bs["total_liabilities"] / bs["total_assets"].replace(0, np.nan)
    bs["avail_month"] = pd.to_datetime(bs["year"].astype(str) + "-04-30")

    factor = bs[["stkcd", "avail_month", "leverage"]].dropna().copy()
    return _merge_annual_factor(universe, factor, "leverage")


def factor_asset_growth(universe):
    bs = _prepare_annual_fundamentals()
    bs["total_assets_lag"] = bs.groupby("stkcd")["total_assets"].shift(1)
    bs["asset_growth"] = (
        (bs["total_assets"] - bs["total_assets_lag"]) / bs["total_assets_lag"].replace(0, np.nan)
    )
    bs["avail_month"] = pd.to_datetime(bs["year"].astype(str) + "-04-30")

    factor = bs[["stkcd", "avail_month", "asset_growth"]].dropna().copy()
    return _merge_annual_factor(universe, factor, "asset_growth")


def factor_accruals(universe):
    bs = _prepare_annual_fundamentals()
    bs["accruals_ratio"] = bs["accruals"] / bs["total_assets"].replace(0, np.nan)
    bs["avail_month"] = pd.to_datetime(bs["year"].astype(str) + "-04-30")

    factor = bs[["stkcd", "avail_month", "accruals_ratio"]].dropna().copy()
    return _merge_annual_factor(universe, factor, "accruals_ratio")


def factor_roa(universe):
    bs = _prepare_annual_fundamentals()
    bs["roa"] = bs["net_income"] / bs["total_assets"].replace(0, np.nan)
    bs["avail_month"] = pd.to_datetime(bs["year"].astype(str) + "-04-30")

    factor = bs[["stkcd", "avail_month", "roa"]].dropna().copy()
    return _merge_annual_factor(universe, factor, "roa")


def _merge_annual_factor(universe, factor_df, factor_name):
    factor_df = factor_df.rename(columns={"avail_month": "avail_start"})
    factor_df["avail_end"] = factor_df["avail_start"] + pd.DateOffset(months=12)

    result_parts = []
    for _, row in factor_df.iterrows():
        mask = (
            (universe["stkcd"] == row["stkcd"])
            & (universe["month"] >= row["avail_start"])
            & (universe["month"] < row["avail_end"])
        )
        matched = universe.loc[mask, ["stkcd", "month"]].copy()
        matched[factor_name] = row[factor_name]
        result_parts.append(matched)

    if not result_parts:
        return pd.DataFrame(columns=["stkcd", "month", factor_name])

    result = pd.concat(result_parts, ignore_index=True)
    result.drop_duplicates(subset=["stkcd", "month"], keep="last", inplace=True)
    return result


def factor_bm(universe):
    bs = _prepare_annual_fundamentals()
    bs["avail_month"] = pd.to_datetime(bs["year"].astype(str) + "-04-30")
    bs_slim = bs[["stkcd", "avail_month", "total_equity"]].dropna().copy()

    factor_df = bs_slim.rename(columns={"avail_month": "avail_start"})
    factor_df["avail_end"] = factor_df["avail_start"] + pd.DateOffset(months=12)

    result_parts = []
    for _, row in factor_df.iterrows():
        mask = (
            (universe["stkcd"] == row["stkcd"])
            & (universe["month"] >= row["avail_start"])
            & (universe["month"] < row["avail_end"])
        )
        matched = universe.loc[mask, ["stkcd", "month", "mkt_cap_total"]].copy()
        matched["bm"] = row["total_equity"] / (matched["mkt_cap_total"].replace(0, np.nan) * 1000)
        result_parts.append(matched[["stkcd", "month", "bm"]])

    if not result_parts:
        return pd.DataFrame(columns=["stkcd", "month", "bm"])

    result = pd.concat(result_parts, ignore_index=True)
    result.drop_duplicates(subset=["stkcd", "month"], keep="last", inplace=True)
    return result.dropna()


def factor_analyst_rating(universe):
    analyst = _load("analyst_rating")

    rank_change_map = {"上调": 1, "维持": 0, "下调": -1, "首次": np.nan}
    analyst["rank_change_num"] = analyst["rank_change"].map(rank_change_map)
    analyst = analyst.dropna(subset=["rank_change_num", "declare_date"]).copy()
    analyst["month"] = analyst["declare_date"].dt.to_period("M").dt.to_timestamp()

    monthly_rating = (
        analyst.groupby(["stkcd", "month"])["rank_change_num"]
        .mean()
        .reset_index()
        .rename(columns={"rank_change_num": "analyst_rating_chg"})
    )

    result = universe[["stkcd", "month"]].merge(
        monthly_rating, on=["stkcd", "month"], how="left"
    )
    return result[["stkcd", "month", "analyst_rating_chg"]].dropna()


def factor_seo(universe):
    seo = _load("seasoned_equity")
    seo = seo.dropna(subset=["listing_date"]).copy()
    seo["month"] = seo["listing_date"].dt.to_period("M").dt.to_timestamp()

    seo_events = seo[["stkcd", "month"]].drop_duplicates()
    seo_events["had_seo"] = 1

    expanded = []
    for _, row in seo_events.iterrows():
        for offset in range(12):
            expanded.append({
                "stkcd": row["stkcd"],
                "month": row["month"] + pd.DateOffset(months=offset),
                "seo_indicator": 1,
            })

    if not expanded:
        return pd.DataFrame(columns=["stkcd", "month", "seo_indicator"])

    seo_panel = pd.DataFrame(expanded)
    seo_panel.drop_duplicates(subset=["stkcd", "month"], inplace=True)

    result = universe[["stkcd", "month"]].merge(
        seo_panel, on=["stkcd", "month"], how="left"
    )
    result["seo_indicator"] = result["seo_indicator"].fillna(0)
    return result[["stkcd", "month", "seo_indicator"]]


def build_all_factors():
    output_path = os.path.join(PROCESSED_DIR, "all_factors.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)

    universe = build_stock_universe()

    factors = [
        factor_size(universe),
        factor_momentum(universe),
        factor_reversal(universe),
        factor_analyst_rating(universe),
        factor_seo(universe),
    ]

    factors.extend(_build_fundamental_factors_vectorized(universe))

    daily_exists = os.path.exists(os.path.join(PROCESSED_DIR, "daily_return.parquet"))
    if daily_exists:
        factors.append(factor_turnover(universe))
        factors.append(factor_idiovol(universe))
    else:
        pass

    panel = universe[["stkcd", "month", "ret", "mkt_cap_total", "mkt_cap_float", "industry"]].copy()
    for factor_df in factors:
        factor_col = [c for c in factor_df.columns if c not in ("stkcd", "month")][0]
        panel = panel.merge(factor_df, on=["stkcd", "month"], how="left")

    panel.to_parquet(output_path, index=False)
    return panel


def _build_fundamental_factors_vectorized(universe):
    bs = _prepare_annual_fundamentals()

    bs["leverage"] = bs["total_liabilities"] / bs["total_assets"].replace(0, np.nan)
    bs["total_assets_lag"] = bs.groupby("stkcd")["total_assets"].shift(1)
    bs["asset_growth"] = (bs["total_assets"] - bs["total_assets_lag"]) / bs["total_assets_lag"].replace(0, np.nan)
    bs["accruals_ratio"] = bs["accruals"] / bs["total_assets"].replace(0, np.nan)
    bs["roa"] = bs["net_income"] / bs["total_assets"].replace(0, np.nan)

    # 年报在次年4月底可用
    bs["avail_year"] = bs["year"] + 1

    universe_copy = universe[["stkcd", "month", "mkt_cap_total"]].copy()
    universe_copy["cal_year"] = universe_copy["month"].dt.year
    universe_copy["cal_month"] = universe_copy["month"].dt.month
    universe_copy["use_year"] = np.where(
        universe_copy["cal_month"] >= 5,
        universe_copy["cal_year"],
        universe_copy["cal_year"] - 1
    )

    bs_slim = bs[["stkcd", "year", "leverage", "asset_growth", "accruals_ratio", "roa", "total_equity"]].copy()
    merged = universe_copy.merge(
        bs_slim, left_on=["stkcd", "use_year"], right_on=["stkcd", "year"], how="left"
    )

    merged["bm"] = merged["total_equity"] / (merged["mkt_cap_total"].replace(0, np.nan) * 1000)

    factor_names = ["leverage", "asset_growth", "accruals_ratio", "roa", "bm"]
    results = []
    for fname in factor_names:
        df = merged[["stkcd", "month", fname]].dropna().copy()
        results.append(df)

    return results


if __name__ == "__main__":
    import time
    start = time.time()
    panel = build_all_factors()
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s")
