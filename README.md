# Replicating McLean & Pontiff (2016) with China A-Share Data

This project replicates the core analysis from McLean and Pontiff (2016, *Journal of Finance*), "Does Academic Research Destroy Stock Return Predictability?", using China A-share market data from CSMAR. The original paper documents that cross-sectional return predictors weaken significantly after academic publication in the US market. We examine whether a similar decay pattern exists in China.

## Data

All data comes from CSMAR (China Stock Market & Accounting Research Database), covering 2000–2026. The raw datasets include:

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

### Coverage

Most factors have >93% coverage over the universe. Exceptions: turnover and idiosyncratic volatility have ~35% coverage because daily return data is only available from 2021 onward; analyst rating change has ~10% coverage (data starts around 2019).

## Portfolio Construction

For each factor, we form **quintile long-short portfolios** each month:

- Rank all stocks by the factor value and assign them to five equal-sized groups (Q1–Q5).
- The long-short return = mean return of the long quintile − mean return of the short quintile.
- The long/short direction depends on the factor's expected sign (e.g., for size, we go long small-cap Q1 and short large-cap Q5; for ROA, we go long high-ROA Q5 and short low-ROA Q1).
- For the binary SEO indicator, we simply compare SEO firms vs. non-SEO firms.
- Months with fewer than 30 stocks per quintile are dropped.

## Sample Periods

Following the original paper's framework, we split the sample into three periods:

| Period | Years | Purpose |
|--------|-------|---------|
| In-sample | 2000–2010 | Original factor discovery period |
| Out-of-sample | 2011–2015 | Pre-publication validation |
| Post-publication | 2016–2026 | Decay analysis after dissemination |

## Experiments and Results

### Table I: Summary Statistics

Average monthly long-short returns (%) across the three periods:

| Factor | Category | In-Sample | t-stat | Post-Publication | t-stat |
|--------|----------|-----------|--------|------------------|--------|
| Reversal | Market | +1.34 | 3.30 | +0.74 | 2.14 |
| ROA | Fundamental | +1.62 | 3.11 | +1.20 | 3.41 |
| Size | Market | −2.31 | −4.11 | −2.88 | −6.50 |
| B/M | Valuation | −3.47 | −8.24 | −3.25 | −7.45 |
| Momentum | Market | −0.88 | −1.60 | −0.31 | −0.78 |
| Accruals | Fundamental | −0.60 | −3.47 | −0.20 | −1.51 |
| Asset growth | Fundamental | −0.68 | −1.84 | −1.11 | −3.65 |
| SEO | Event | −0.70 | −1.73 | +0.29 | 2.31 |
| Leverage | Fundamental | +0.00 | 0.01 | +0.04 | 0.17 |

Cross-factor average in-sample monthly return: **−0.63%**, post-publication: **−1.69%**.

### Table II: Panel Regression

We estimate the panel regression with predictor fixed effects:

$$R_{i,t} = \alpha_i + \beta_1 \cdot \text{PostSample}_t + \beta_2 \cdot \text{PostPublication}_t + \varepsilon_{i,t}$$

where \(R_{i,t}\) is the long-short return of factor \(i\) in month \(t\), and standard errors are clustered by month.

| Variable | Coefficient | Clustered SE | t-stat |
|----------|-------------|-------------|--------|
| Post-Sample (β₁) | +0.054 | 0.153 | 0.35 |
| Post-Publication (β₂) | +0.102 | 0.130 | 0.79 |

Neither coefficient is statistically significant, suggesting that on average, factor returns in China did not experience a sharp structural break at the period boundaries. This contrasts with the US results where McLean & Pontiff find significant negative β₂.

### Table III: Cross-Sectional Decay by Factor

| Factor | IS Mean (%) | PP Mean (%) | Decay (pp) | Decay (%) |
|--------|-------------|-------------|------------|-----------|
| SEO | −0.70 | +0.29 | +0.99 | −141% (sign flip) |
| Momentum | −0.88 | −0.31 | +0.56 | −64% |
| Accruals | −0.60 | −0.20 | +0.41 | −67% |
| Reversal | +1.34 | +0.74 | −0.60 | −45% |
| ROA | +1.62 | +1.20 | −0.42 | −26% |
| B/M | −3.47 | −3.25 | +0.21 | −6% |
| Size | −2.31 | −2.88 | −0.56 | +24% (strengthened) |
| Asset growth | −0.68 | −1.11 | −0.43 | +63% (strengthened) |

Some factors (reversal, momentum, accruals) show decay consistent with the publication effect. Others (size, asset growth) actually strengthened post-publication.

### Table IV: Decay by Predictor Type

| Category | N Factors | IS Mean (%) | PP Mean (%) | Decay (pp) |
|----------|-----------|-------------|-------------|------------|
| Market | 5 | −0.62 | −2.22 | −1.61 |
| Fundamental | 4 | +0.08 | −0.02 | −0.10 |
| Valuation | 1 | −3.47 | −3.25 | +0.21 |
| Event | 1 | −0.70 | +0.29 | +0.99 |

Market-based factors show the largest post-publication change (−1.61 pp), but this is driven by turnover and idiosyncratic volatility entering the post-publication sample (they lack in-sample data). Fundamental and valuation factors are relatively stable.

### Table V: Arbitrage Costs

We proxy arbitrage costs by the average size percentile rank of stocks in the extreme quintiles. Factors involving smaller stocks (lower size rank ≈ higher arbitrage cost) are expected to decay less, because arbitrage is harder.

The regression of decay on average size rank yields mixed results, consistent with the fact that China's short-selling constraints and retail-dominated market structure make the arbitrage cost channel less clean than in the US.

## Comparison with McLean & Pontiff (2016) US Results

| Dimension | US (Original Paper) | China (This Study) |
|-----------|--------------------|--------------------|
| **Average IS return** | ~0.45% per month | −0.63% per month |
| **Post-publication decay** | ~32% decline | Mixed: some factors decay, others strengthen |
| **Panel regression β₂** | Significantly negative (−0.12, t≈−3) | Insignificant (+0.10, t=0.79) |
| **Strongest decay** | Earnings-based, event-driven factors | Momentum, accruals, reversal |
| **Factors that persist** | Factors with high arbitrage costs | Size, B/M, asset growth |

### Why the Differences?

1. **Market structure**: China's A-share market is dominated by retail investors (~80% of trading volume), whereas the US is institution-dominated. Retail investors are less likely to read academic papers and trade on published anomalies, so the "publication effect" is weaker.

2. **Short-selling constraints**: China introduced short-selling only in 2010 (margin trading pilot), and the eligible stock list remains limited. Many anomalies require shorting to exploit, so even if investors learn about them, implementation is restricted.

3. **Regulatory environment**: China's market has frequent trading halts, IPO suspensions, and policy-driven interventions that can dominate factor returns, making it harder to isolate the publication effect.

4. **Factor definitions**: Some factors (e.g., size, B/M) reflect structural features of the Chinese market (state-owned enterprise premiums, shell value of small caps) rather than pure mispricing, so they persist regardless of academic attention.

5. **Sample period**: Our post-publication period (2016–2026) includes several major market events (2015 crash aftermath, COVID-19, regulatory crackdowns on tech/education sectors) that may mask or amplify factor returns independently of publication effects.

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

## Usage

```bash
pip install -r requirements.txt

python src/data_loader.py
python src/factor_builder.py
python src/portfolio.py
python src/regression.py
python src/visualization.py
```

Results are saved to `output/`.
