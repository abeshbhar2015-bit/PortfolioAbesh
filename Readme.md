# NIFTY 50 Portfolio Optimisation: MVP vs HRP

A Streamlit application comparing a long-only Minimum Variance Portfolio (MVP) with Hierarchical Risk Parity (HRP).

## Default setup

- Price field: Yahoo Finance `Close` only (not `Adj Close`).
- Requested period: 1 January 2025 to 31 October 2026.
- Training period: 1 January 2025 to 30 April 2026.
- Out-of-sample test period: 1 May 2026 to 31 October 2026.
- Default universe: Reliance Industries, Bharti Airtel, HDFC Bank, ICICI Bank, State Bank of India, TCS, Bajaj Finance, Larsen & Toubro, Hindustan Unilever and Infosys.
- Weights are estimated once from training-period returns and held constant during the test period.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

6. The app opens in your browser. Keep the default dates and select **Run portfolio analysis**.

## Method overview

- Daily simple returns are calculated from the `Close` price series.
- MVP minimises portfolio variance under long-only and fully-invested constraints.
- HRP clusters stocks using correlation distance, orders the hierarchy and allocates risk recursively through bisection.
- Out-of-sample metrics include total return, CAGR, annualised volatility, Sharpe ratio and maximum drawdown.
- Charts include cumulative test returns, risk-return comparison, drawdowns and allocation pies.
- Allocation weights and performance results can be downloaded as CSV.

## Important limitations

- This is an educational backtest, not investment advice.
- Yahoo Finance must be reachable when the app downloads data.
- Close-only analysis excludes dividend income. The provider may restate historical close prices for stock splits.
- No transaction costs, taxes, bid-ask spread, slippage or periodic rebalancing are modelled.
- The default universe reflects a market-cap snapshot from November 2026, so the constituent choice is not point-in-time for the earlier training window. For a strict historical backtest, replace the universe with constituents known at the training start date.
- The final observed test date may be the last trading session on or before 31 October 2026 (31 October 2026 is a Saturday).
