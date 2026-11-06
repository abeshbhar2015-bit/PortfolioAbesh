"""Streamlit app: Minimum Variance Portfolio (MVP) vs Hierarchical Risk Parity (HRP).

Data: Yahoo Finance daily `Close` prices only (not `Adj Close`).
Default period: 2025-01-01 to 2026-10-31.
Default test period: final six calendar months (2026-05-01 to 2026-10-31).
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# Top 10 NIFTY 50 companies by market capitalisation, using the 4-Nov-2026
# snapshot listed by SectorAlpha. Symbols are editable in the sidebar.
DEFAULT_TICKERS = [
    "RELIANCE.NS",
    "BHARTIARTL.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "TCS.NS",
    "BAJFINANCE.NS",
    "LT.NS",
    "HINDUNILVR.NS",
    "INFY.NS",
]

TRADING_DAYS = 252


@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_close_prices(
    tickers: Tuple[str, ...], start_date: date, end_date: date
) -> pd.DataFrame:
    """Download daily Close prices. end_date is inclusive for the user."""
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    raw = yf.download(
        tickers=list(tickers),
        start=pd.Timestamp(start_date),
        end=end_exclusive,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no price data for the selected period.")

    # yfinance may return either (Price field, ticker) or (ticker, Price field)
    # MultiIndex columns, depending on version/settings.
    if isinstance(raw.columns, pd.MultiIndex):
        close_level = None
        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                close_level = level
                break
        if close_level is None:
            raise ValueError("The downloaded data does not contain a 'Close' field.")
        close = raw.xs("Close", axis=1, level=close_level, drop_level=True)
    else:
        if "Close" not in raw.columns:
            raise ValueError("The downloaded data does not contain a 'Close' field.")
        close = raw[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close.columns = [str(col) for col in close.columns]
    missing = [ticker for ticker in tickers if ticker not in close.columns]
    if missing:
        raise ValueError(
            "Price data is missing for: " + ", ".join(missing)
            + ". Check the ticker symbols or try again later."
        )

    close = close.loc[:, list(tickers)].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close.apply(pd.to_numeric, errors="coerce")
    # Use common trading dates so both portfolios are tested on identical days.
    close = close.dropna(axis=0, how="any")
    close = close[(close.index >= pd.Timestamp(start_date)) & (close.index <= pd.Timestamp(end_date))]

    if close.empty:
        raise ValueError("No complete common Close-price observations remain after cleaning.")
    return close


def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not calculate valid portfolio weights.")
    return weights / total


def minimum_variance_weights(train_returns: pd.DataFrame) -> pd.Series:
    """Long-only, fully invested global minimum-variance weights."""
    cov = train_returns.cov().to_numpy(dtype=float)
    # A tiny diagonal ridge improves numerical stability without changing the
    # interpretation of the objective.
    cov = cov + np.eye(cov.shape[0]) * 1e-10
    n_assets = cov.shape[0]

    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    def variance_gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * cov @ weights

    result = minimize(
        portfolio_variance,
        x0=np.repeat(1.0 / n_assets, n_assets),
        jac=variance_gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 3000, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"MVP optimisation failed: {result.message}")
    weights = _normalise_weights(result.x)
    return pd.Series(weights, index=train_returns.columns, name="MVP")


def _cluster_variance(covariance: pd.DataFrame, cluster_items: Sequence[str]) -> float:
    """Variance of an inverse-variance-weighted cluster portfolio."""
    cov_slice = covariance.loc[list(cluster_items), list(cluster_items)].to_numpy(dtype=float)
    diagonal = np.diag(cov_slice)
    inverse_variance = 1.0 / np.maximum(diagonal, 1e-12)
    cluster_weights = inverse_variance / inverse_variance.sum()
    return float(cluster_weights @ cov_slice @ cluster_weights)


def hierarchical_risk_parity_weights(train_returns: pd.DataFrame) -> pd.Series:
    """Calculate long-only HRP weights using correlation clustering and bisection."""
    covariance = train_returns.cov()
    correlation = train_returns.corr().clip(-1.0, 1.0)

    # Correlation distance used by the standard HRP construction.
    distance = np.sqrt(np.clip((1.0 - correlation.to_numpy()) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    condensed_distance = squareform(distance, checks=False)
    tree = linkage(condensed_distance, method="single", optimal_ordering=True)
    ordered_positions = leaves_list(tree)
    ordered_assets = [train_returns.columns[i] for i in ordered_positions]

    weights = pd.Series(1.0, index=ordered_assets, dtype=float)
    clusters: List[List[str]] = [ordered_assets]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left_cluster = cluster[:split]
        right_cluster = cluster[split:]

        left_variance = _cluster_variance(covariance, left_cluster)
        right_variance = _cluster_variance(covariance, right_cluster)
        denominator = left_variance + right_variance
        left_allocation = 0.5 if denominator <= 0 else 1.0 - left_variance / denominator

        weights.loc[left_cluster] *= left_allocation
        weights.loc[right_cluster] *= 1.0 - left_allocation
        clusters.extend([left_cluster, right_cluster])

    weights = weights.reindex(train_returns.columns)
    weights = pd.Series(_normalise_weights(weights.to_numpy()), index=train_returns.columns, name="HRP")
    return weights


def portfolio_metrics(
    daily_returns: pd.Series, annual_risk_free_rate: float
) -> Dict[str, float]:
    """Calculate out-of-sample performance metrics from daily portfolio returns."""
    daily_returns = daily_returns.dropna().astype(float)
    if daily_returns.empty:
        raise ValueError("The test period does not contain enough daily returns.")

    wealth = (1.0 + daily_returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    years = len(daily_returns) / TRADING_DAYS
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else np.nan
    annual_volatility = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    annual_mean_return = float(daily_returns.mean() * TRADING_DAYS)
    annual_rf_daily_basis = annual_risk_free_rate
    sharpe = (
        (annual_mean_return - annual_rf_daily_basis) / annual_volatility
        if annual_volatility > 0
        else np.nan
    )
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "Total Return": total_return,
        "Annualised Return (CAGR)": cagr,
        "Annualised Volatility": annual_volatility,
        "Sharpe Ratio": float(sharpe),
        "Maximum Drawdown": max_drawdown,
    }


def make_wealth_series(daily_returns: pd.Series) -> pd.Series:
    return (1.0 + daily_returns.fillna(0.0)).cumprod() - 1.0


def format_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2%}"


st.set_page_config(
    page_title="NIFTY 50 | MVP vs HRP Portfolio Optimisation",
    page_icon="📈",
    layout="wide",
)

st.title("NIFTY 50 Portfolio Optimisation")
st.subheader("Minimum Variance Portfolio (MVP) vs Hierarchical Risk Parity (HRP)")
st.write(
    "Build portfolio weights using the training period, then compare both strategies "
    "on the final six-month out-of-sample test period. Only daily **Close** prices are used."
)

with st.sidebar:
    st.header("Model settings")
    start_date = st.date_input("Data start date", value=date(2025, 1, 1))
    end_date = st.date_input("Data end date", value=date(2026, 10, 31))
    test_months = st.number_input("Number of test months", min_value=1, max_value=12, value=6, step=1)
    risk_free_pct = st.number_input(
        "Annual risk-free rate (%) for Sharpe ratio",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.25,
    )
    ticker_text = st.text_area(
        "NIFTY 50 ticker symbols (Yahoo Finance format)",
        value=", ".join(DEFAULT_TICKERS),
        height=150,
        help="Edit the universe if required. Use the .NS suffix for NSE-listed shares.",
    )
    run_button = st.button("Run portfolio analysis", type="primary", use_container_width=True)

if "analysis_has_run" not in st.session_state:
    st.session_state.analysis_has_run = False

if run_button:
    st.session_state.analysis_has_run = True

if not st.session_state.analysis_has_run:
    st.info("Review the settings in the sidebar, then select **Run portfolio analysis**.")
    st.markdown(
        "**Default universe:** Reliance Industries, Bharti Airtel, HDFC Bank, ICICI Bank, "
        "State Bank of India, TCS, Bajaj Finance, Larsen & Toubro, Hindustan Unilever and Infosys."
    )
    st.stop()

try:
    tickers = tuple(dict.fromkeys(t.strip().upper() for t in ticker_text.split(",") if t.strip()))
    if len(tickers) < 2:
        raise ValueError("Enter at least two valid ticker symbols.")
    if start_date >= end_date:
        raise ValueError("The data start date must be earlier than the data end date.")
    if test_months < 1:
        raise ValueError("The test period must be at least one month.")

    test_start = (pd.Timestamp(end_date) - pd.DateOffset(months=int(test_months)) + pd.Timedelta(days=1)).date()
    if test_start <= start_date:
        raise ValueError("The selected test period leaves too little training data. Reduce test months or move the start date earlier.")

    with st.spinner("Downloading Close prices and optimising portfolios..."):
        prices = download_close_prices(tickers, start_date, end_date)
        daily_returns = prices.pct_change().dropna(how="any")
        train_returns = daily_returns.loc[daily_returns.index < pd.Timestamp(test_start)]
        test_returns = daily_returns.loc[daily_returns.index >= pd.Timestamp(test_start)]

        if len(train_returns) < 60:
            raise ValueError("Fewer than 60 training return observations are available. Expand the training period.")
        if len(test_returns) < 20:
            raise ValueError("Fewer than 20 test return observations are available. Check the end date and ticker data.")

        mvp_weights = minimum_variance_weights(train_returns)
        hrp_weights = hierarchical_risk_parity_weights(train_returns)

        mvp_test_returns = test_returns @ mvp_weights
        hrp_test_returns = test_returns @ hrp_weights
        annual_rf = float(risk_free_pct) / 100.0
        mvp_stats = portfolio_metrics(mvp_test_returns, annual_rf)
        hrp_stats = portfolio_metrics(hrp_test_returns, annual_rf)

    st.success("Analysis completed successfully.")

    training_end = train_returns.index.max().date()
    actual_test_start = test_returns.index.min().date()
    actual_test_end = test_returns.index.max().date()

    info_cols = st.columns(4)
    info_cols[0].metric("Stocks analysed", f"{len(tickers)}")
    info_cols[1].metric("Training observations", f"{len(train_returns):,}")
    info_cols[2].metric("Test observations", f"{len(test_returns):,}")
    info_cols[3].metric("Test window", f"{actual_test_start:%d %b %Y} – {actual_test_end:%d %b %Y}")

    with st.expander("Methodology and date split", expanded=True):
        st.markdown(
            f"""
            - **Price field:** Yahoo Finance `Close` only; `Adj Close`, Open, High, Low and Volume are not used.
            - **Requested data window:** {start_date:%d %b %Y} to {end_date:%d %b %Y} (end date inclusive).
            - **Training window:** {start_date:%d %b %Y} to {training_end:%d %b %Y}.
            - **Out-of-sample test window:** {actual_test_start:%d %b %Y} to {actual_test_end:%d %b %Y}.
            - **Returns:** daily simple percentage changes in Close prices. The first test-day return is measured from the immediately preceding available close, even when that prior close is in the training window.
            - **MVP:** long-only, fully invested minimum-variance optimisation using the training-period covariance matrix.
            - **HRP:** correlation-distance hierarchical clustering, single linkage, quasi-diagonal ordering and recursive bisection.
            - **Sharpe ratio:** annualised mean daily excess return divided by annualised daily volatility; the risk-free rate is editable in the sidebar.
            - **Trading-day convention:** 252 days per year.
            """
        )

    st.header("Out-of-sample performance")
    metric_table = pd.DataFrame(
        {"MVP": mvp_stats, "HRP": hrp_stats}
    ).T
    display_metrics = metric_table.copy()
    for column in ["Total Return", "Annualised Return (CAGR)", "Annualised Volatility", "Maximum Drawdown"]:
        display_metrics[column] = display_metrics[column].map(format_pct)
    display_metrics["Sharpe Ratio"] = display_metrics["Sharpe Ratio"].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
    st.dataframe(display_metrics, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.metric("MVP Sharpe Ratio", f"{mvp_stats['Sharpe Ratio']:.3f}" if np.isfinite(mvp_stats['Sharpe Ratio']) else "—")
    with right:
        st.metric("HRP Sharpe Ratio", f"{hrp_stats['Sharpe Ratio']:.3f}" if np.isfinite(hrp_stats['Sharpe Ratio']) else "—")

    wealth_df = pd.DataFrame(
        {
            "MVP": make_wealth_series(mvp_test_returns) * 100.0,
            "HRP": make_wealth_series(hrp_test_returns) * 100.0,
        }
    )
    wealth_fig = go.Figure()
    wealth_fig.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df["MVP"], mode="lines", name="MVP"))
    wealth_fig.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df["HRP"], mode="lines", name="HRP"))
    wealth_fig.update_layout(
        title="Cumulative out-of-sample return",
        xaxis_title="Date",
        yaxis_title="Cumulative return (%)",
        hovermode="x unified",
        legend_title="Strategy",
    )
    st.plotly_chart(wealth_fig, use_container_width=True)

    risk_return_fig = go.Figure()
    for strategy, stats in [("MVP", mvp_stats), ("HRP", hrp_stats)]:
        risk_return_fig.add_trace(
            go.Scatter(
                x=[stats["Annualised Volatility"] * 100.0],
                y=[stats["Annualised Return (CAGR)"] * 100.0],
                mode="markers+text",
                text=[strategy],
                textposition="top center",
                marker={"size": 15},
                name=strategy,
                hovertemplate=(
                    f"{strategy}<br>Annualised volatility: "
                    "%{x:.2f}%<br>Annualised return: %{y:.2f}%<extra></extra>"
                ),
            )
        )
    risk_return_fig.update_layout(
        title="Risk–return comparison (test period)",
        xaxis_title="Annualised volatility / risk (%)",
        yaxis_title="Annualised return – CAGR (%)",
        legend_title="Strategy",
    )
    st.plotly_chart(risk_return_fig, use_container_width=True)

    drawdown_df = pd.DataFrame(
        {
            "MVP": ((1.0 + mvp_test_returns).cumprod() / (1.0 + mvp_test_returns).cumprod().cummax() - 1.0) * 100.0,
            "HRP": ((1.0 + hrp_test_returns).cumprod() / (1.0 + hrp_test_returns).cumprod().cummax() - 1.0) * 100.0,
        }
    )
    drawdown_fig = go.Figure()
    drawdown_fig.add_trace(go.Scatter(x=drawdown_df.index, y=drawdown_df["MVP"], mode="lines", name="MVP"))
    drawdown_fig.add_trace(go.Scatter(x=drawdown_df.index, y=drawdown_df["HRP"], mode="lines", name="HRP"))
    drawdown_fig.update_layout(
        title="Out-of-sample drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        legend_title="Strategy",
    )
    st.plotly_chart(drawdown_fig, use_container_width=True)

    st.header("Portfolio allocations")
    alloc_left, alloc_right = st.columns(2)
    mvp_alloc = mvp_weights.sort_values(ascending=False).rename_axis("Ticker").reset_index(name="Weight")
    hrp_alloc = hrp_weights.sort_values(ascending=False).rename_axis("Ticker").reset_index(name="Weight")

    with alloc_left:
        mvp_pie = px.pie(mvp_alloc, names="Ticker", values="Weight", hole=0.28, title="MVP allocation")
        mvp_pie.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(mvp_pie, use_container_width=True)
    with alloc_right:
        hrp_pie = px.pie(hrp_alloc, names="Ticker", values="Weight", hole=0.28, title="HRP allocation")
        hrp_pie.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(hrp_pie, use_container_width=True)

    allocation_table = pd.DataFrame(
        {
            "Ticker": list(tickers),
            "MVP Weight": mvp_weights.reindex(tickers).values,
            "HRP Weight": hrp_weights.reindex(tickers).values,
        }
    )
    allocation_table["MVP Weight (%)"] = allocation_table["MVP Weight"] * 100.0
    allocation_table["HRP Weight (%)"] = allocation_table["HRP Weight"] * 100.0
    allocation_table["Difference (HRP - MVP, pp)"] = allocation_table["HRP Weight (%)"] - allocation_table["MVP Weight (%)"]
    allocation_display = allocation_table[
        ["Ticker", "MVP Weight (%)", "HRP Weight (%)", "Difference (HRP - MVP, pp)"]
    ].copy()
    st.dataframe(
        allocation_display.style.format(
            {
                "MVP Weight (%)": "{:.2f}%",
                "HRP Weight (%)": "{:.2f}%",
                "Difference (HRP - MVP, pp)": "{:+.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Download key outputs for use in a report or assignment.
    summary_download = metric_table.copy()
    allocation_download = allocation_table.set_index("Ticker")
    st.download_button(
        "Download performance and allocation results (CSV)",
        data=pd.concat(
            [summary_download.add_prefix("Metric - ").T, allocation_download.T.add_prefix("Allocation - ")],
            axis=1,
        ).to_csv().encode("utf-8"),
        file_name="nifty50_mvp_hrp_results.csv",
        mime="text/csv",
    )

    st.caption(
        "Interpretation caution: these are historical, long-only, static-weight backtests. "
        "They exclude transaction costs, taxes, slippage and dividend income. The default stock universe "
        "is selected using a 4-Nov-2026 market-cap snapshot, so it is not a point-in-time constituent "
        "selection for the 2025–May 2026 training window."
    )

except Exception as exc:
    st.error(f"Unable to complete the analysis: {exc}")
    st.caption("Check the ticker symbols, internet connection and selected dates, then run the analysis again.")
