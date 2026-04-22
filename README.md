# Replicating McLean & Pontiff (2016) with China A-Share Data

This project replicates the core analysis from McLean and Pontiff (2016, *Journal of Finance*), "Does Academic Research Destroy Stock Return Predictability?", using China A-share market data from CSMAR. The original paper documents that cross-sectional return predictors weaken significantly after academic publication in the US market. We examine whether a similar decay pattern exists in China.

## Data

All data comes from CSMAR (China Stock Market & Accounting Research Database), covering 1991–2026. The raw datasets include:

| Dataset | Records | Description |
|---------|---------|-------------|
| Monthly stock returns | 800,045 | Return, market cap, industry, listing date |
| Daily stock returns | 5,700,111 | Price, volume, daily return |
| Balance sheet | 325,825 | Total assets, liabilities, equity, net income, accruals |
| Analyst ratings | 961,070 | Rating changes (upgrade/maintain/downgrade) |
| Monthly quotes | 1,695,200 | Adjusted close prices |
| SEO events | 5,863 | Seasoned equity offering dates |

### Data Processing

1. **Format conversion**: Raw CSMAR `.xlsx` files are parsed (skipping Chinese/English header rows via `skiprows=[1,2]`), columns are renamed to English, and saved as `.parquet` for fast I/O.
2. **Stock universe construction**: We keep only A-share stocks (market type ∈ {1, 4, 16, 32}), exclude IPOs listed less than 6 months, and drop observations with missing returns. The final universe contains **757,647 stock-month observations** across **5,049 unique stocks**.
3. **Stock codes** are standardized to 6-digit zero-padded strings (e.g., `"000002"`).

## Factor Construction

We construct 12 anomaly variables grouped into four categories. All factors are computed monthly.

### Market-based factors (5)

| Factor | Construction |
|--------|-------------|
| **Size** | Log of total market capitalization |
| **Momentum** | Cumulative log return over months t−12 to t−2 (skipping the most recent month) |
| **Short-term reversal** | Prior month's return |
| **Turnover** | Average daily turnover ratio (volume / float market cap) over the past 3 months |
| **Idiosyncratic volatility** | Standard deviation of residuals from regressing daily stock returns on equal-weighted market returns within each month (minimum 10 trading days required) |

### Fundamental factors (4)

| Factor | Construction |
|--------|-------------|
| **Leverage** | Total liabilities / total assets |
| **Asset growth** | (Total assets − lagged total assets) / lagged total assets |
| **Accruals ratio** | Accruals / total assets |
| **ROA** | Net income / total assets |

Fundamental factors use annual report data (fiscal year-end = December). To avoid look-ahead bias, annual reports are assumed available after April 30 of the following year: months May–December use the current year's report, months January–April use the prior year's report.

### Valuation factor (1)

| Factor | Construction |
|--------|-------------|
| **Book-to-market (B/M)** | Total equity / (total market cap × 1000), where market cap is in thousands of RMB |

### Event-based factors (2)

| Factor | Construction |
|--------|-------------|
| **Analyst rating change** | Monthly average of analyst rating changes (upgrade = +1, maintain = 0, downgrade = −1) |
| **SEO indicator** | Binary variable = 1 if the firm conducted a seasoned equity offering in the past 12 months |

### Data Availability and Coverage

Most factors (size, momentum, reversal, leverage, asset growth, accruals, ROA, B/M, SEO) have >93% coverage over the universe and span the full data period from the early 1990s onward.

However, three factors have severely limited temporal coverage due to the availability of underlying CSMAR raw data:

| Factor | Required Raw Data | Actual Data Available | Portfolio Data Range | Impact |
|--------|------------------|----------------------|---------------------|--------|
| **Idio. Vol.** | Daily stock returns | 2021–2025 only | 2021–2025 (60 months) | Published in 2006; should have IS (1991–2000) and OOS (2001–2006), but daily data starts too late — entire sample is post-publication |
| **Turnover** | Daily trading volume | 2021–2025 only | 2021–2025 (59 months) | Published in 1998; same issue — no pre-publication data available |
| **Analyst Rating** | Analyst rating records | 2019–2026 only | No portfolio data (insufficient coverage per month) | Published in 2004; should have IS (1991–1998) and OOS (1999–2004), but rating data starts in 2019 — too late and too sparse to form portfolios |

As a result, only **Asset Growth** (published 2008, data from 1994) has complete in-sample, out-of-sample, and post-publication periods in the Chinese data. If earlier daily return data were available from CSMAR, Idiosyncratic Volatility could also provide a full three-period analysis.

## Portfolio Construction

For each factor, we form **quintile long-short portfolios** each month:

- Rank all stocks by the factor value and assign them to five equal-sized groups (Q1–Q5).
- The long-short return = mean return of the long quintile − mean return of the short quintile.
- The long/short direction depends on the factor's expected sign (e.g., for size, we go long small-cap Q1 and short large-cap Q5; for ROA, we go long high-ROA Q5 and short low-ROA Q1).
- For the binary SEO indicator, we simply compare SEO firms vs. non-SEO firms.
- Months with fewer than 30 stocks per quintile are dropped.

## Sample Periods

Following the original paper's framework, each factor has its own three-period split based on the **real publication date** of the original academic paper that discovered it (from McLean & Pontiff 2016, Table IA.III):

| Factor | Original Paper | Sample Period | Published | In-Sample | Out-of-Sample | Post-Publication |
|--------|---------------|---------------|-----------|-----------|---------------|-----------------|
| Size | Banz (JFE 1981) | 1926–1975 | 1981 | — | — | 1991–2026 |
| Momentum | Jegadeesh & Titman (JF 1993) | 1964–1989 | 1993 | — | — | 1994–2026 |
| Reversal | Jegadeesh (1990) | 1934–1987 | 1990 | — | — | 1991–2026 |
| Turnover | Datar et al. (JFM 1998) | 1962–1991 | 1998 | — | — | 1999–2026 |
| Idio. Vol. | Ang et al. (JF 2006) | 1986–2000 | 2006 | 1991–2000 | 2001–2006 | 2007–2026 |
| Leverage | Bhandari (JFE 1988) | 1946–1981 | 1988 | — | — | 1991–2026 |
| Asset Growth | Cooper et al. (JF 2008) | 1968–2003 | 2008 | 1991–2003 | 2004–2008 | 2009–2026 |
| Accruals | Sloan (AR 1996) | 1962–1991 | 1996 | — | — | 1997–2026 |
| ROA | Haugen & Baker (JFE 1996) | 1979–1993 | 1996 | — | 1994–1996 | 1997–2026 |
| B/M | Fama & French (JF 1992) | 1963–1990 | 1992 | — | — | 1993–2026 |
| Analyst Rating | Jegadeesh et al. (JF 2004) | 1985–1998 | 2004 | 1991–1998 | 1999–2004 | 2005–2026 |
| SEO | Loughran & Ritter (JF 1995) | 1975–1984 | 1995 | — | — | 1996–2026 |

Since China's A-share data begins in 1991, most factors' original sample periods ended before the Chinese market data starts. For these factors, the entire Chinese sample falls into the **post-publication** period — meaning Chinese investors already had access to the published research throughout our data window. Only Asset Growth (pub. 2008) has all three periods (IS, OOS, PP) in the Chinese data. ROA (pub. 1996) has out-of-sample (1994–1996, 32 months) and post-publication data but no in-sample data.

## Experiments and Results

### Table I: Summary Statistics

This table shows the average monthly long-short returns across all three periods (In-Sample, Out-of-Sample, Post-Publication) for each factor.

| Factor | Category | Pub Year | IS Mean (%) | IS N | OOS Mean (%) | OOS N | PP Mean (%) | PP N |
|--------|----------|----------|-------------|------|--------------|-------|-------------|------|
| Asset Growth | Fundamental | 2008 | −1.12 | 116 | −0.90 | 60 | −1.00 | 207 |
| ROA | Fundamental | 1996 | — | 0 | +2.38 | 32 | +1.36 | 351 |
| Reversal | Market | 1990 | — | 0 | — | 0 | +1.20 | 389 |
| Leverage | Fundamental | 1988 | — | 0 | — | 0 | +0.13 | 390 |
| SEO | Event | 1995 | — | 0 | — | 0 | −0.25 | 322 |
| Accruals | Fundamental | 1996 | — | 0 | — | 0 | −0.39 | 335 |
| Momentum | Market | 1993 | — | 0 | — | 0 | −0.66 | 379 |
| Turnover | Market | 1998 | — | 0 | — | 0 | −2.11 | 59 |
| Size | Market | 1981 | — | 0 | — | 0 | −2.32 | 390 |
| B/M | Valuation | 1992 | — | 0 | — | 0 | −4.16 | 390 |
| Idio. Vol. | Market | 2006 | — | 0 | — | 0 | −10.97 | 60 |

**Key observations:**
- **Asset Growth** is the only factor with complete IS → OOS → PP data. The returns are: IS = −1.12%, OOS = −0.90%, PP = −1.00%. The decay from IS to PP is only −10.9%, suggesting minimal post-publication weakening.
- **ROA** has OOS and PP data but no IS data. The decline from OOS (+2.38%) to PP (+1.36%) is notable, but the OOS period is short (32 months).
- All other factors have data only in the post-publication period because their original papers were published before Chinese market data begins (1991).

### Table II: Panel Regression

We estimate the panel regression with predictor fixed effects, where period dummies are assigned per-factor based on each factor's real publication date:

$$R_{i,t} = \alpha_i + \beta_1 \cdot \text{PostSample}_{i,t} + \beta_2 \cdot \text{PostPublication}_{i,t} + \varepsilon_{i,t}$$

Where:
- **PostSample** = 1 for Out-of-Sample period (after original sample ends, before publication)
- **PostPublication** = 1 for Post-Publication period (after publication date)

Since only Asset Growth has all three periods in the Chinese data, the regression is effectively based on one factor:

| Variable | Coefficient | Clustered SE | t-stat |
|----------|-------------|-------------|--------|
| Post-Sample (β₁) | +0.218 | 0.852 | 0.26 |
| Post-Publication (β₂) | +0.122 | 0.526 | 0.23 |

**Interpretation:**
- **β₁ (Post-Sample)**: The coefficient is positive (+0.218%) but not statistically significant (t = 0.26). This suggests no significant decay from IS to OOS period.
- **β₂ (Post-Publication)**: The coefficient is also positive (+0.122%) and not statistically significant (t = 0.23). This suggests no significant decay from IS to PP period.

Neither coefficient is statistically significant, suggesting **no meaningful post-publication decay** for Asset Growth in China. This contrasts with the US results where McLean & Pontiff (2016) found significant negative coefficients.

### Table III: Cross-Sectional Decay by Factor

This table shows the complete decay pattern from In-Sample → Out-of-Sample → Post-Publication for each factor.

| Factor | IS Mean (%) | IS N | OOS Mean (%) | OOS N | PP Mean (%) | PP N | Decay IS→OOS (%) | Decay IS→PP (%) | Decay OOS→PP (%) |
|--------|-------------|------|--------------|-------|-------------|------|------------------|-----------------|------------------|
| Asset Growth | −1.12 | 116 | −0.90 | 60 | −1.00 | 207 | −19.4% | −10.9% | +10.6% |
| ROA | — | 0 | +2.38 | 32 | +1.36 | 351 | — | — | −43.0% |

**Key findings:**
- **Asset Growth** is the only factor with complete IS → OOS → PP data. The decay from IS to OOS is −19.4%, but from OOS to PP it actually increases by +10.6%, suggesting no systematic post-publication decay.
- **ROA** has OOS and PP data but no IS data. The decline from OOS (+2.38%) to PP (+1.36%) is −43.0%, but this is based on only 32 months of OOS data.
- Other factors have no IS or OOS data because their original papers were published before Chinese market data begins (1991).

The limited cross-sectional variation (only one factor with complete data) prevents meaningful cross-sectional analysis of decay patterns.

### Key Insights

1. **Structural limitation**: Most US-discovered anomalies were already published before Chinese market data begins (1991), so the entire Chinese sample is post-publication for most factors. This fundamentally limits our ability to directly replicate the IS vs. PP comparison.

2. **Asset Growth factor behaves differently in China**: For Asset Growth (the only factor with complete IS → OOS → PP data), we find:
   - **IS Mean**: −1.12% (negative, opposite to US +0.58%)
   - **OOS Mean**: −0.90%
   - **PP Mean**: −1.00%
   - **IS → OOS change**: +19.4% (returns increase, not decay)
   - **OOS → PP change**: +10.6% (returns increase, not decay)
   - **Overall IS → PP change**: +10.9% (returns increase, not decay)

3. **Contrast with US results**: McLean & Pontiff (2016) found:
   - US IS Mean: +0.582%, OOS Mean: +0.402%, PP Mean: +0.264%
   - US IS → PP decay: −58% (significant decline)
   - US Panel regression β₂: −0.337% (t = −3.74, significant)

   In China, we find:
   - China IS Mean: −1.12%, OOS Mean: −0.90%, PP Mean: −1.00%
   - China IS → PP change: +10.9% (returns increase, not decay)
   - China Panel regression β₂: +0.122% (t = 0.23, not significant)

4. **Interpretation**: The opposite sign and lack of decay suggest that:
   - **Asset Growth anomaly works differently in China**: The factor produces negative returns in China, opposite to the positive returns in the US.
   - **Academic publication has weaker effect in China**: Unlike the US where publication leads to significant decay, Chinese returns do not decay post-publication.
   - **Possible explanations**:
     - **Retail-dominated market**: China's A-share market is dominated by retail investors (~80% of trading volume), who are less likely to read academic papers and trade on published anomalies.
     - **Short-selling constraints**: China introduced short-selling only in 2010, and the eligible stock list remains limited. Many anomalies require shorting to exploit.
     - **Different market structure**: China's market has different institutional features (state-owned enterprises, policy interventions) that may affect factor returns differently.
     - **Information asymmetry**: Academic research may take longer to diffuse to Chinese investors compared to US investors.

## Comparison with McLean & Pontiff (2016) US Results

| Dimension | US (Original Paper) | China (This Study) |
|-----------|--------------------|--------------------|
| **Number of predictors** | 97 | 12 |
| **Data period** | 1926–2013 | 1991–2026 |
| **Factors with IS+OOS+PP data** | 97 (all) | 1 (Asset Growth) |
| **Factors with OOS+PP data** | 97 (all) | 2 (Asset Growth, ROA) |
| **Average IS return** | ~0.45% per month | −1.12% (Asset Growth) |
| **Average OOS return** | ~0.30% per month | −0.90% (Asset Growth), +2.38% (ROA) |
| **Average PP return** | ~0.30% per month | −1.00% (Asset Growth), +1.36% (ROA) |
| **IS→OOS decay** | ~30% decline | −19.4% (Asset Growth) |
| **IS→PP decay** | ~32% decline | −10.9% (Asset Growth) |
| **OOS→PP decay** | ~10% decline | +10.6% (Asset Growth), −43.0% (ROA) |
| **Panel regression β₁ (OOS)** | Significantly negative | Insignificant (+0.22, t=0.26) |
| **Panel regression β₂ (PP)** | Significantly negative (−0.32, t≈−3) | Insignificant (+0.12, t=0.23) |

### Why the Differences?

1. **Temporal mismatch**: Most US anomalies were published before Chinese market data begins (1991). The original papers used US data from the 1920s–1990s, so by the time Chinese data is available, these factors are already in the post-publication regime. This fundamentally limits the in-sample vs. post-publication comparison.

2. **Market structure**: China's A-share market is dominated by retail investors (~80% of trading volume), whereas the US is institution-dominated. Retail investors are less likely to read academic papers and trade on published anomalies, so the "publication effect" is weaker.

3. **Short-selling constraints**: China introduced short-selling only in 2010 (margin trading pilot), and the eligible stock list remains limited. Many anomalies require shorting to exploit, so even if investors learn about them, implementation is restricted.

4. **Regulatory environment**: China's market has frequent trading halts, IPO suspensions, and policy-driven interventions that can dominate factor returns, making it harder to isolate the publication effect.

5. **Factor definitions**: Some factors (e.g., size, B/M) reflect structural features of the Chinese market (state-owned enterprise premiums, shell value of small caps) rather than pure mispricing, so they persist regardless of academic attention.

## Project Structure

```
├── data/
│   ├── raw/              # Raw CSMAR xlsx files
│   └── processed/        # Cleaned parquet files
├── src/
│   ├── data_loader.py    # CSMAR data parsing and parquet conversion
│   ├── factor_builder.py # 12 factor construction
│   ├── portfolio.py      # Quintile long-short portfolio formation
│   ├── regression.py     # Panel regressions (Tables I–V)
│   └── visualization.py  # Figures 1–5
├── output/               # Generated CSV tables and PNG figures
├── materials/            # Reference paper PDF
└── requirements.txt
```

## Data Download

The raw data files are too large (~2.5 GB) to include in the repository. You need to download them from CSMAR and place them in the `data/raw/` directory with the following structure:

```
data/
└── raw/
    ├── analyst_rating/     # Analyst rating records (.xlsx)
    ├── balance_sheet/      # Balance sheet data (.xlsx)
    ├── daily_return/       # Daily stock returns (.xlsx)
    ├── monthly_quote/      # Monthly adjusted close prices (.xlsx)
    ├── monthly_return/     # Monthly stock returns (.xlsx)
    ├── rights_issue/       # Rights issue events (.xlsx)
    └── seasoned_equity/    # SEO events (.xlsx)
```

After placing the raw `.xlsx` files, run `python src/data_loader.py` to parse and convert them into `.parquet` format under `data/processed/`. All subsequent scripts read from `data/processed/`.

## Usage

```bash
pip install -r requirements.txt

# Step 1: Parse raw CSMAR xlsx files → parquet
python src/data_loader.py

# Step 2: Construct 12 anomaly factors
python src/factor_builder.py

# Step 3: Form quintile long-short portfolios
python src/portfolio.py

# Step 4: Run panel regressions (Tables I–V)
python src/regression.py

# Step 5: Generate figures
python src/visualization.py
```

Results are saved to `output/`.
