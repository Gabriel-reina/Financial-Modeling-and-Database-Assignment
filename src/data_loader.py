import os
import glob
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def standardize_stkcd(series):
    return series.astype(str).str.strip().str.zfill(6)

def load_monthly_return():
    output_path = os.path.join(PROCESSED_DIR, "monthly_return.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)
    filepath = os.path.join(RAW_DIR, "monthly_return", "TRD_Mnth(Merge Query).xlsx")
    df = pd.read_excel(filepath, header=0, skiprows=[1, 2])

    column_mapping = {
        "TRD_Mnth.Stkcd": "stkcd",
        "TRD_Mnth.Trdmnt": "month",
        "TRD_Mnth.Msmvosd": "mkt_cap_float",
        "TRD_Mnth.Msmvttl": "mkt_cap_total",
        "TRD_Mnth.Mretwd": "ret",
        "TRD_Mnth.Markettype": "market_type",
        "csmar_listedcoinfo.Nnindnme": "industry",
        "csmar_listedcoinfo.Listdt": "list_date",
    }
    df.rename(columns=column_mapping, inplace=True)
    df["stkcd"] = standardize_stkcd(df["stkcd"])
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df.sort_values(["stkcd", "month"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    return df


def load_daily_return():
    output_path = os.path.join(PROCESSED_DIR, "daily_return.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)
    daily_dir = os.path.join(RAW_DIR, "daily_return")
    xlsx_files = sorted(glob.glob(os.path.join(daily_dir, "TRD_Dalyr*.xlsx")))
    xlsx_files = [f for f in xlsx_files if "[DES]" not in f]

    all_dfs = []
    for filepath in xlsx_files:
        df_chunk = pd.read_excel(filepath, header=0, skiprows=[1, 2])
        all_dfs.append(df_chunk)

    df = pd.concat(all_dfs, ignore_index=True)

    column_mapping = {
        "Stkcd": "stkcd",
        "Trddt": "date",
        "Opnprc": "open",
        "Hiprc": "high",
        "Loprc": "low",
        "Clsprc": "close",
        "Dnshrtrd": "volume",
        "Dnvaltrd": "turnover_value",
        "Dsmvosd": "mkt_cap_float",
        "Dsmvtll": "mkt_cap_total",
        "Dretwd": "ret",
        "ChangeRatio": "change_ratio",
    }
    df.rename(columns=column_mapping, inplace=True)
    df.drop(columns=["Ahshrtrd_D", "Ahvaltrd_D"], errors="ignore", inplace=True)

    df["stkcd"] = standardize_stkcd(df["stkcd"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(["stkcd", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(subset=["stkcd", "date"], inplace=True)

    df.to_parquet(output_path, index=False)
    return df


def load_balance_sheet():
    output_path = os.path.join(PROCESSED_DIR, "balance_sheet.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)
    filepath = os.path.join(RAW_DIR, "balance_sheet", "FS_Combas(Merge Query).xlsx")
    df = pd.read_excel(filepath, header=0, skiprows=[1, 2])

    column_mapping = {
        "FS_Combas.Stkcd": "stkcd",
        "FS_Combas.ShortName": "short_name",
        "FS_Combas.Accper": "report_date",
        "FS_Combas.Typrep": "report_type",
        "FS_Combas.A001000000": "total_assets",
        "FS_Combas.A002101000": "short_term_debt",
        "FS_Combas.A002125000": "current_portion_lt_debt",
        "FS_Combas.A002201000": "long_term_debt",
        "FS_Combas.A002203000": "bonds_payable",
        "FS_Combas.A002204000": "long_term_payable",
        "FS_Combas.A002000000": "total_liabilities",
        "FS_Combas.A003000000": "total_equity",
        "FS_Comins.B001100000": "revenue",
        "FS_Comins.B001216000": "rd_expense",
        "FS_Comins.B001300000": "operating_profit",
        "FS_Comins.B002000000": "net_income",
        "FS_Comins.B002000101": "net_income_parent",
        "FI_T8.F082101B": "accruals",
        "FS_Comscfd.C003006000": "dividend_paid",
        "csmar_listedcoinfo.Nnindnme": "industry",
        "csmar_listedcoinfo.IndnmeZX": "industry_zx",
    }
    df.rename(columns=column_mapping, inplace=True)
    df["stkcd"] = standardize_stkcd(df["stkcd"])
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    df = df[df["report_type"] == "A"].copy()
    df.sort_values(["stkcd", "report_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    return df


def load_analyst_rating():
    output_path = os.path.join(PROCESSED_DIR, "analyst_rating.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)
    filepath = os.path.join(RAW_DIR, "analyst_rating", "AF_Bench.xlsx")
    df = pd.read_excel(filepath, header=0, skiprows=[1, 2])

    column_mapping = {
        "Stkcd": "stkcd",
        "ReportID": "report_id",
        "Rptdt": "report_date",
        "DeclareDate": "declare_date",
        "Ananm": "analyst_name",
        "Brokern": "broker",
        "Investrank": "invest_rank",
        "Stdrank": "std_rank",
        "Rankchg": "rank_change",
    }
    df.rename(columns=column_mapping, inplace=True)
    keep_cols = [c for c in column_mapping.values() if c in df.columns]
    df = df[keep_cols].copy()

    df["stkcd"] = standardize_stkcd(df["stkcd"])
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["declare_date"] = pd.to_datetime(df["declare_date"], errors="coerce")
    df.sort_values(["stkcd", "declare_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    return df


def load_monthly_quote():
    output_path = os.path.join(PROCESSED_DIR, "monthly_quote.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)
    quote_dir = os.path.join(RAW_DIR, "monthly_quote")
    xlsx_files = sorted(glob.glob(os.path.join(quote_dir, "TRD_BwardQuotationMonth*.xlsx")))
    xlsx_files = [f for f in xlsx_files if "[DES]" not in f]

    all_dfs = []
    for filepath in xlsx_files:
        df_chunk = pd.read_excel(filepath, header=0, skiprows=[1, 2])
        all_dfs.append(df_chunk)

    df = pd.concat(all_dfs, ignore_index=True)

    column_mapping = {
        "TradingMonth": "month",
        "Symbol": "stkcd",
        "CloseDate": "close_date",
        "ClosePrice": "close_price_adj",
    }
    df.rename(columns=column_mapping, inplace=True)
    df["stkcd"] = standardize_stkcd(df["stkcd"])
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["close_date"] = pd.to_datetime(df["close_date"], errors="coerce")
    df.sort_values(["stkcd", "month", "close_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(output_path, index=False)
    return df


def load_rights_issue():
    output_path = os.path.join(PROCESSED_DIR, "rights_issue.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)
    filepath = os.path.join(RAW_DIR, "rights_issue", "RS_Robasic.xlsx")
    df = pd.read_excel(filepath, header=0, skiprows=[1, 2])

    column_mapping = {
        "Stkcd": "stkcd",
        "Roadt": "announce_date",
        "Tlstdt": "listing_date",
    }
    df.rename(columns=column_mapping, inplace=True)
    df = df[["stkcd", "announce_date", "listing_date"]].copy()
    df["stkcd"] = standardize_stkcd(df["stkcd"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")

    df.to_parquet(output_path, index=False)
    return df


def load_seasoned_equity():
    output_path = os.path.join(PROCESSED_DIR, "seasoned_equity.parquet")
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)
    filepath = os.path.join(RAW_DIR, "seasoned_equity", "RS_Aibasic.xlsx")
    df = pd.read_excel(filepath, header=0, skiprows=[1, 2])

    column_mapping = {
        "Stkcd": "stkcd",
        "Aitype": "seo_type",
        "Ailtdt": "listing_date",
        "Aistdt": "issue_date",
    }
    df.rename(columns=column_mapping, inplace=True)
    keep_cols = [c for c in column_mapping.values() if c in df.columns]
    df = df[keep_cols].copy()
    df["stkcd"] = standardize_stkcd(df["stkcd"])
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")

    df.to_parquet(output_path, index=False)
    return df


def load_all():
    datasets = {}
    datasets["monthly_return"] = load_monthly_return()
    datasets["balance_sheet"] = load_balance_sheet()
    datasets["analyst_rating"] = load_analyst_rating()
    datasets["monthly_quote"] = load_monthly_quote()
    datasets["rights_issue"] = load_rights_issue()
    datasets["seasoned_equity"] = load_seasoned_equity()
    datasets["daily_return"] = load_daily_return()
    return datasets


if __name__ == "__main__":
    import time
    start = time.time()
    data = load_all()
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s")