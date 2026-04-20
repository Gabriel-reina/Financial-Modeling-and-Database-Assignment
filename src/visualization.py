import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IN_SAMPLE_END = "2010-12-31"
OUT_SAMPLE_END = "2015-12-31"

FACTOR_LABELS = {
    "size": "Size",
    "momentum": "Momentum",
    "reversal": "Reversal",
    "turnover": "Turnover",
    "idiovol": "Idio. Vol.",
    "leverage": "Leverage",
    "asset_growth": "Asset Growth",
    "accruals_ratio": "Accruals",
    "roa": "ROA",
    "bm": "B/M",
    "analyst_rating_chg": "Analyst Rating",
    "seo_indicator": "SEO",
}


def load_portfolio_returns():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "portfolio_returns.parquet"))
    df["month"] = pd.to_datetime(df["month"])
    return df


def figure1_cumulative_returns():
    port_ret = load_portfolio_returns()

    valid_factors = []
    for factor_name in port_ret["factor_name"].unique():
        sub = port_ret[port_ret["factor_name"] == factor_name]
        if len(sub) >= 60:
            valid_factors.append(factor_name)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=False)
    axes = axes.flatten()

    plot_idx = 0
    for factor_name in sorted(valid_factors):
        if plot_idx >= 9:
            break

        sub = port_ret[port_ret["factor_name"] == factor_name].sort_values("month")
        sub["cum_return"] = (1 + sub["ls_return"]).cumprod() - 1

        ax = axes[plot_idx]
        ax.plot(sub["month"], sub["cum_return"] * 100, linewidth=1.2, color="#2166ac")

        for date, label, color in [
            (pd.Timestamp(IN_SAMPLE_END), "End of Sample", "#d73027"),
            (pd.Timestamp(OUT_SAMPLE_END), "Publication", "#fc8d59"),
        ]:
            if sub["month"].min() < date < sub["month"].max():
                ax.axvline(x=date, color=color, linestyle="--", alpha=0.7, linewidth=1)

        ax.set_title(FACTOR_LABELS.get(factor_name, factor_name), fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)", fontsize=9)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.2)
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Figure 1: Cumulative Long-Short Portfolio Returns\n(China A-Shares)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "figure1_cumulative_returns.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def figure2_return_comparison():
    port_ret = load_portfolio_returns()

    periods = {
        "In-Sample\n(2000-2010)": ("2000-01-01", IN_SAMPLE_END),
        "Out-of-Sample\n(2011-2015)": ("2011-01-01", OUT_SAMPLE_END),
        "Post-Publication\n(2016-2026)": ("2016-01-01", "2026-12-31"),
    }

    valid_factors = []
    for factor_name in port_ret["factor_name"].unique():
        sub = port_ret[port_ret["factor_name"] == factor_name]
        in_sample = sub[(sub["month"] >= "2000-01-01") & (sub["month"] <= IN_SAMPLE_END)]
        if len(in_sample) >= 24:
            valid_factors.append(factor_name)

    factor_labels = [FACTOR_LABELS.get(f, f) for f in sorted(valid_factors)]
    x_positions = np.arange(len(valid_factors))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["#2166ac", "#fc8d59", "#d73027"]
    for i, (period_name, (start, end)) in enumerate(periods.items()):
        means = []
        for factor_name in sorted(valid_factors):
            sub = port_ret[
                (port_ret["factor_name"] == factor_name)
                & (port_ret["month"] >= start)
                & (port_ret["month"] <= end)
            ]
            means.append(sub["ls_return"].mean() * 100 if len(sub) > 0 else 0)

        ax.bar(x_positions + i * bar_width, means, bar_width,
               label=period_name, color=colors[i], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Factor", fontsize=12)
    ax.set_ylabel("Average Monthly L/S Return (%)", fontsize=12)
    ax.set_title("Figure 2: Average Monthly Long-Short Returns by Period\n(China A-Shares)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels(factor_labels, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "figure2_return_comparison.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def figure3_decay_scatter():
    port_ret = load_portfolio_returns()
    from scipy import stats

    factor_stats = []
    for factor_name in port_ret["factor_name"].unique():
        sub = port_ret[port_ret["factor_name"] == factor_name]
        in_sample = sub[(sub["month"] >= "2000-01-01") & (sub["month"] <= IN_SAMPLE_END)]
        post_pub = sub[sub["month"] > OUT_SAMPLE_END]

        if len(in_sample) < 12 or len(post_pub) < 12:
            continue

        is_mean = in_sample["ls_return"].mean() * 100
        pp_mean = post_pub["ls_return"].mean() * 100
        decay = pp_mean - is_mean

        factor_stats.append({
            "factor": factor_name,
            "is_mean": is_mean,
            "decay": decay,
        })

    df_stats = pd.DataFrame(factor_stats)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(df_stats["is_mean"], df_stats["decay"],
               s=100, c="#2166ac", alpha=0.8, edgecolors="white", linewidth=1.5, zorder=5)

    for _, row in df_stats.iterrows():
        ax.annotate(FACTOR_LABELS.get(row["factor"], row["factor"]),
                     (row["is_mean"], row["decay"]),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=9, alpha=0.8)

    if len(df_stats) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_stats["is_mean"], df_stats["decay"]
        )
        x_line = np.linspace(df_stats["is_mean"].min() - 0.5, df_stats["is_mean"].max() + 0.5, 100)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color="#d73027", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"OLS: slope={slope:.3f}, R²={r_value**2:.3f}")

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("In-Sample Mean Monthly Return (%)", fontsize=12)
    ax.set_ylabel("Return Decay: Post-Pub − In-Sample (%)", fontsize=12)
    ax.set_title("Figure 3: In-Sample Returns vs. Post-Publication Decay\n(China A-Shares)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "figure3_decay_scatter.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def figure4_decay_by_type():
    table4 = pd.read_csv(os.path.join(OUTPUT_DIR, "table4_by_predictor_type.csv"))

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = table4["Category"].values
    x_positions = np.arange(len(categories))
    bar_width = 0.35

    ax.bar(x_positions - bar_width / 2, table4["IS_Mean(%)"], bar_width,
           label="In-Sample (2000-2010)", color="#2166ac", alpha=0.85, edgecolor="white")
    ax.bar(x_positions + bar_width / 2, table4["PP_Mean(%)"], bar_width,
           label="Post-Publication (2016-2026)", color="#d73027", alpha=0.85, edgecolor="white")

    ax.set_xlabel("Predictor Type", fontsize=12)
    ax.set_ylabel("Average Monthly L/S Return (%)", fontsize=12)
    ax.set_title("Figure 4: Returns by Predictor Type: In-Sample vs. Post-Publication\n(China A-Shares)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "figure4_decay_by_type.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def figure5_arbitrage_cost():
    table5 = pd.read_csv(os.path.join(OUTPUT_DIR, "table5_arbitrage_costs.csv"))
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(table5["avg_size_rank"], table5["decay"],
               s=100, c="#2166ac", alpha=0.8, edgecolors="white", linewidth=1.5, zorder=5)

    for _, row in table5.iterrows():
        ax.annotate(FACTOR_LABELS.get(row["factor"], row["factor"]),
                     (row["avg_size_rank"], row["decay"]),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=9, alpha=0.8)

    if len(table5) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            table5["avg_size_rank"], table5["decay"]
        )
        x_line = np.linspace(table5["avg_size_rank"].min() - 0.02,
                             table5["avg_size_rank"].max() + 0.02, 100)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color="#d73027", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"OLS: slope={slope:.2f}, R²={r_value**2:.3f}, p={p_value:.3f}")

    ax.set_xlabel("Average Size Rank of Portfolio Stocks\n(Higher = Larger Stocks = Lower Arbitrage Cost)",
                  fontsize=11)
    ax.set_ylabel("Return Decay: Post-Pub − In-Sample (%)", fontsize=12)
    ax.set_title("Figure 5: Arbitrage Costs and Post-Publication Return Decay\n(China A-Shares)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "figure5_arbitrage_cost.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def generate_all_figures():
    figure1_cumulative_returns()
    figure2_return_comparison()
    figure3_decay_scatter()
    figure4_decay_by_type()
    figure5_arbitrage_cost()



if __name__ == "__main__":
    generate_all_figures()
