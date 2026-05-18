
"""
Time Series Forecasting App — Mountain Path Academy edition
-----------------------------------------------------------
Run:
    pip install streamlit pandas numpy plotly
    streamlit run streamlit_forecast_app.py

Features
--------
- Ships with a default synthetic quarterly dataset so the app is immediately usable.
- Upload your own CSV (date column + value column).
- Frequency selector: Daily, Weekly, Monthly, Quarterly, Annual (sets seasonal period s).
- Trend estimator: Centred Moving Average (CMA) or Linear Regression.
- Decomposition: Additive or Multiplicative (or run BOTH and compare).
- Out-of-sample forecast horizon configurable.
- Error metrics: MAE / MAD / MAPE / RMSE on the in-sample fit and on a hold-out.
- Plotly charts: raw series with trend, detrended series, seasonal indices,
  fit + forecast band, residuals.
"""

from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Theme — Mountain Path Academy inspired
# ---------------------------------------------------------------------------
MPA_PINE = "#1F4D3F"
MPA_SLATE = "#3B4A55"
MPA_SAND = "#C49E62"
MPA_SAND_LT = "#F6EFDE"
MPA_PARCHMENT = "#FCFAF4"
MPA_PINE_LT = "#D0E1D9"
MPA_INK = "#1C1F22"
MPA_SUBTLE = "#787876"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=MPA_PARCHMENT,
        plot_bgcolor=MPA_PARCHMENT,
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=MPA_INK),
        title=dict(font=dict(size=16, color=MPA_PINE)),
        xaxis=dict(gridcolor="#E6E2D6", linecolor=MPA_SLATE, zerolinecolor="#E6E2D6"),
        yaxis=dict(gridcolor="#E6E2D6", linecolor=MPA_SLATE, zerolinecolor="#E6E2D6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=50, b=40),
    )
)

st.set_page_config(
    page_title="Mountain Path · Time Series Forecasting",
    page_icon="⛰️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {MPA_PARCHMENT}; }}
      .mpa-hero {{
        border-left: 4px solid {MPA_SAND};
        padding: 10px 18px;
        background: {MPA_SAND_LT};
        border-radius: 4px;
        margin-bottom: 12px;
      }}
      .mpa-hero h1 {{
        margin: 0 0 4px 0;
        color: {MPA_PINE};
        font-weight: 700;
      }}
      .mpa-hero p {{
        margin: 0;
        color: {MPA_SLATE};
        font-size: 0.95rem;
      }}
      .metric-card {{
        background: white;
        border: 1px solid #E6E2D6;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
      }}
      .metric-card h3 {{
        color: {MPA_SLATE};
        margin: 0;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      .metric-card .val {{
        color: {MPA_PINE};
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
      }}
      div[data-testid="stSidebar"] {{ background-color: {MPA_SAND_LT}; }}
      h2 {{ color: {MPA_PINE}; }}
      h3 {{ color: {MPA_SLATE}; }}
    </style>
    <div class="mpa-hero">
      <h1>Time Series Forecasting — Practitioner's Lab</h1>
      <p>Decompose your series into trend and seasonality, forecast with CMA or Linear Regression, and read off MAE / MAD / MAPE / RMSE.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Forecasting engine
# ---------------------------------------------------------------------------
FREQ_TO_PERIOD = {
    "Daily (s = 7, weekly seasonality)": ("D", 7),
    "Daily (s = 365, annual seasonality)": ("D", 365),
    "Weekly (s = 52)": ("W", 52),
    "Monthly (s = 12)": ("M", 12),
    "Quarterly (s = 4)": ("Q", 4),
    "Annual (no seasonality)": ("A", 1),
}


def centred_moving_average(y: pd.Series, s: int) -> pd.Series:
    if s <= 1:
        return y.copy()
    y = y.astype(float)
    if s % 2 == 1:
        return y.rolling(window=s, center=True).mean()
    ma = y.rolling(window=s, center=False).mean()
    cma = (ma + ma.shift(-1)) / 2.0
    cma = cma.shift(-(s // 2 - 1))
    return cma


def linear_regression_trend(y: pd.Series) -> tuple[pd.Series, float, float]:
    y = y.astype(float)
    t = np.arange(1, len(y) + 1, dtype=float)
    tm, ym = t.mean(), y.mean()
    b = ((t - tm) * (y.values - ym)).sum() / ((t - tm) ** 2).sum()
    a = ym - b * tm
    return pd.Series(a + b * t, index=y.index), a, b


def detrend(y: pd.Series, trend: pd.Series, model: str) -> pd.Series:
    return y - trend if model == "additive" else y / trend


def seasonal_indices(detrended: pd.Series, s: int, model: str,
                     season_of: np.ndarray) -> pd.Series:
    df = pd.DataFrame({"d": detrended.values, "k": season_of}).dropna()
    if df.empty or s <= 1:
        return pd.Series([0.0 if model == "additive" else 1.0] * max(s, 1),
                         index=range(max(s, 1)))
    raw = df.groupby("k")["d"].mean().reindex(range(s))
    raw = raw.fillna(raw.mean())
    if model == "additive":
        return raw - raw.mean()
    return raw * (s / raw.sum())


def recompose(trend: pd.Series, idx: pd.Series, model: str,
              season_of: np.ndarray) -> pd.Series:
    aligned = idx.reindex(season_of).values
    if model == "additive":
        return pd.Series(trend.values + aligned, index=trend.index)
    return pd.Series(trend.values * aligned, index=trend.index)


def error_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    a = actual.astype(float)
    f = forecast.astype(float)
    mask = (~a.isna()) & (~f.isna())
    a, f = a[mask], f[mask]
    if len(a) == 0:
        return {"MAE": np.nan, "MAD": np.nan, "RMSE": np.nan, "MAPE_%": np.nan, "n": 0}
    err = a - f
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    nz = a != 0
    mape = (err[nz].abs() / a[nz].abs()).mean() * 100 if nz.any() else np.nan
    return {"MAE": mae, "MAD": mae, "RMSE": rmse, "MAPE_%": mape, "n": int(mask.sum())}


def extend_index(idx: pd.Index, n_extra: int) -> pd.Index:
    """Extend a DatetimeIndex / PeriodIndex / RangeIndex by n_extra periods.

    Robust to indices whose frequency cannot be inferred (uploaded data with
    near-regular but not exact spacing): we fall back to the median delta.
    """
    if n_extra <= 0:
        return idx

    if isinstance(idx, pd.PeriodIndex):
        return pd.period_range(idx[0], periods=len(idx) + n_extra, freq=idx.freq)

    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or pd.infer_freq(idx)
        if freq is not None:
            try:
                return pd.date_range(idx[0], periods=len(idx) + n_extra, freq=freq)
            except Exception:
                freq = None
        # fallback: use median spacing — works for any near-regular series
        if len(idx) >= 2:
            deltas = np.diff(idx.view("int64"))
            step_ns = int(np.median(deltas))
            step = pd.Timedelta(step_ns, unit="ns")
            future = pd.DatetimeIndex(
                [idx[-1] + step * (k + 1) for k in range(n_extra)]
            )
            return idx.append(future)
        return idx

    # Numeric / RangeIndex fallback
    try:
        start = int(idx[0])
    except Exception:
        start = 0
    return pd.RangeIndex(start=start, stop=start + len(idx) + n_extra)


def run_pipeline(y: pd.Series, s: int, trend_method: str, model: str,
                 horizon: int) -> dict:
    season_of = np.arange(len(y)) % max(s, 1)

    if trend_method == "CMA" and s > 1:
        trend = centred_moving_average(y, s)
        # backfill the CMA tails using a linear extrapolation so we can detrend everywhere
        trend_full = trend.interpolate(method="linear", limit_direction="both")
    else:
        trend_full, a, b = linear_regression_trend(y)
        trend = trend_full.copy()

    detrended = detrend(y, trend_full, model)
    idx = seasonal_indices(detrended, s, model, season_of)
    fitted = recompose(trend_full, idx, model, season_of)
    residuals = y - fitted

    # ---- future forecast
    full_idx = extend_index(y.index, horizon)
    future_idx = full_idx[len(y):]
    season_of_full = np.arange(len(full_idx)) % max(s, 1)

    if trend_method == "CMA" and s > 1:
        # extrapolate trend by fitting a short linear model to the last `s` CMA points
        tail = trend.dropna().tail(s)
        if len(tail) >= 2:
            t_tail = np.arange(len(tail))
            slope, intercept = np.polyfit(t_tail, tail.values, 1)
            t_future = np.arange(len(tail), len(tail) + horizon)
            future_trend_vals = intercept + slope * t_future
        else:
            future_trend_vals = np.repeat(trend.iloc[-1], horizon)
    else:
        a, b = trend_full.iloc[0], (trend_full.iloc[-1] - trend_full.iloc[0]) / (len(trend_full) - 1)
        future_trend_vals = np.array([trend_full.iloc[-1] + b * (k + 1) for k in range(horizon)])

    season_future = season_of_full[len(y):]
    s_aligned_future = idx.reindex(season_future).values
    if model == "additive":
        future_forecast = future_trend_vals + s_aligned_future
    else:
        future_forecast = future_trend_vals * s_aligned_future

    fitted_full = pd.Series(
        np.concatenate([fitted.values, future_forecast]),
        index=full_idx, name="forecast"
    )
    trend_full_ext = pd.Series(
        np.concatenate([trend_full.values, future_trend_vals]),
        index=full_idx, name="trend"
    )

    return {
        "trend": trend_full_ext,
        "fitted": fitted_full,
        "in_sample_fit": fitted,
        "detrended": detrended,
        "seasonal_index": idx,
        "residuals": residuals,
        "metrics": error_metrics(y, fitted),
        "season_of_full": season_of_full,
        "full_index": full_idx,
        "n_train": len(y),
    }


# ---------------------------------------------------------------------------
# Default dataset
# ---------------------------------------------------------------------------
@st.cache_data
def default_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    quarters = pd.period_range("2018Q1", "2025Q4", freq="Q")
    t = np.arange(1, len(quarters) + 1)
    trend_true = 100 + 4.5 * t
    seasonal_mult = np.tile([1.10, 0.95, 0.90, 1.05], len(quarters) // 4)
    noise = rng.normal(0, 4, len(quarters))
    sales = trend_true * seasonal_mult + noise
    df = pd.DataFrame({
        "date": quarters.to_timestamp(),
        "value": np.round(sales, 2),
    })
    return df


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### Data source")
    source = st.radio(
        "Data source",
        ["Default quarterly sales (synthetic)", "Upload CSV or Excel"],
        label_visibility="collapsed",
    )

    uploaded = None
    if source == "Upload CSV or Excel":
        uploaded = st.file_uploader(
            "CSV / XLSX / XLS — must have a date column and a value column",
            type=["csv", "xlsx", "xls"],
            help=("CSV or Excel file. For multi-sheet workbooks you'll get a "
                  "sheet picker. Then map which column is the date and "
                  "which is the value."),
        )

    st.markdown("---")
    st.markdown("### Series settings")
    freq_label = st.selectbox(
        "Sampling frequency / seasonality",
        list(FREQ_TO_PERIOD.keys()),
        index=4,  # Quarterly default
    )
    _, default_s = FREQ_TO_PERIOD[freq_label]
    s = st.number_input("Seasonal period s (override)", min_value=1, max_value=400,
                        value=int(default_s), step=1,
                        help="Number of observations in one full seasonal cycle.")

    st.markdown("---")
    st.markdown("### Forecast model")
    trend_method = st.selectbox("Trend estimator",
                                ["Centred Moving Average (CMA)", "Linear Regression"])
    trend_method_short = "CMA" if "CMA" in trend_method else "LR"

    model_choice = st.radio("Decomposition",
                            ["Additive", "Multiplicative", "Compare both"],
                            horizontal=False)

    horizon = st.slider("Forecast horizon (periods ahead)",
                        min_value=1, max_value=max(8, int(default_s) * 3),
                        value=max(4, int(default_s)))

    holdout_pct = st.slider("Hold-out for back-test (% of history)",
                            min_value=0, max_value=40, value=20, step=5,
                            help="The last N% of data is hidden from the fit and used to score MAPE/MAE out-of-sample.")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def _read_uploaded(file) -> pd.DataFrame:
    """Read an uploaded CSV / XLSX / XLS into a DataFrame.

    For multi-sheet workbooks shows a sheet selector. Returns a flat
    DataFrame; column selection happens downstream.
    """
    name = (file.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)

    if name.endswith((".xlsx", ".xls")):
        engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
        try:
            xls = pd.ExcelFile(file, engine=engine)
        except ImportError as e:
            st.error(
                f"Reading {name.split('.')[-1].upper()} files requires the "
                f"`{engine}` package. Install it with `pip install {engine}` "
                f"and restart the app.  ({e})"
            )
            st.stop()
        sheet_names = xls.sheet_names
        if len(sheet_names) == 1:
            return xls.parse(sheet_names[0])
        sheet = st.selectbox("Excel sheet", sheet_names, index=0,
                             help="Workbook has multiple sheets — pick one.")
        return xls.parse(sheet)

    # Fallback: try CSV
    file.seek(0)
    return pd.read_csv(file)


if source == "Upload CSV or Excel" and uploaded is not None:
    try:
        df_raw = _read_uploaded(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    if df_raw.empty or len(df_raw.columns) < 2:
        st.error("Uploaded file must have at least two columns "
                 "(a date column and a value column).")
        st.stop()

    cols = list(df_raw.columns)
    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox("Date column", cols, index=0)
    with c2:
        default_val_idx = 1 if len(cols) > 1 else 0
        # If the obvious 2nd column is non-numeric, try to pick the first numeric one
        for i, c in enumerate(cols):
            if c != date_col and pd.api.types.is_numeric_dtype(df_raw[c]):
                default_val_idx = i
                break
        value_col = st.selectbox("Value column", cols, index=default_val_idx)

    df = df_raw[[date_col, value_col]].copy()
    df.columns = ["date", "value"]
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        st.warning("Date column could not be parsed as datetime; "
                   "using row index as the time axis.")
        df["date"] = pd.RangeIndex(start=0, stop=len(df))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    if df.empty:
        st.error("After parsing, the value column had no numeric data.")
        st.stop()

    with st.expander("Preview uploaded data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
else:
    df = default_dataset()

if len(df) < max(2 * s, 8):
    st.warning(f"Series is short ({len(df)} obs). Forecast quality with s={s} will be poor; "
               "consider a smaller seasonal period or more history.")

y = pd.Series(df["value"].values, index=pd.DatetimeIndex(df["date"]) if pd.api.types.is_datetime64_any_dtype(df["date"]) else df["date"], name="value")
y = y.sort_index()

# Hold-out split
n_hold = int(round(len(y) * holdout_pct / 100))
y_train = y.iloc[: len(y) - n_hold] if n_hold > 0 else y
y_test = y.iloc[len(y) - n_hold:] if n_hold > 0 else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Run the models
# ---------------------------------------------------------------------------
def _run(model: str) -> dict:
    fc_horizon = horizon + n_hold
    out = run_pipeline(y_train, s=int(s), trend_method=trend_method_short,
                       model=model, horizon=fc_horizon)
    # back-test metrics on the hold-out, indexed positionally so the labels
    # of the rebuilt forecast timeline don't need to match y_test's index
    if n_hold > 0 and not y_test.empty:
        n_train = out["n_train"]
        # the first n_hold points of the future portion correspond to y_test
        f_hold_vals = out["fitted"].iloc[n_train: n_train + len(y_test)].values
        f_hold = pd.Series(f_hold_vals, index=y_test.index, name="forecast")
        out["holdout_forecast"] = f_hold
        out["holdout_metrics"] = error_metrics(y_test, f_hold)
    else:
        out["holdout_forecast"] = pd.Series(dtype=float)
        out["holdout_metrics"] = {"MAE": np.nan, "MAD": np.nan, "RMSE": np.nan,
                                  "MAPE_%": np.nan, "n": 0}
    return out


if model_choice == "Additive":
    runs = {"additive": _run("additive")}
elif model_choice == "Multiplicative":
    runs = {"multiplicative": _run("multiplicative")}
else:
    runs = {"additive": _run("additive"), "multiplicative": _run("multiplicative")}


# ---------------------------------------------------------------------------
# Header: KPIs
# ---------------------------------------------------------------------------
st.markdown("### Headline accuracy")

model_for_kpi = list(runs.keys())[0]
m_in = runs[model_for_kpi]["metrics"]
m_out = runs[model_for_kpi]["holdout_metrics"]

def kpi_card(label: str, value, fmt="{:,.3f}") -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value))):
        v = "—"
    else:
        v = fmt.format(value)
    return f"<div class='metric-card'><h3>{label}</h3><div class='val'>{v}</div></div>"


cols = st.columns(4)
labels = [("MAE / MAD (fit)", m_in["MAE"], "{:,.3f}"),
          ("MAPE % (fit)", m_in["MAPE_%"], "{:.2f}%"),
          ("MAE / MAD (hold-out)", m_out["MAE"], "{:,.3f}"),
          ("MAPE % (hold-out)", m_out["MAPE_%"], "{:.2f}%")]
for col, (lbl, val, fmt) in zip(cols, labels):
    fmt_str = "{:.2f}" if "%" in fmt else fmt
    if "%" in fmt:
        body = (f"<div class='metric-card'><h3>{lbl}</h3>"
                f"<div class='val'>{val:.2f}%</div></div>"
                if val is not None and not (isinstance(val, float) and np.isnan(val))
                else f"<div class='metric-card'><h3>{lbl}</h3><div class='val'>—</div></div>")
    else:
        body = kpi_card(lbl, val)
    col.markdown(body, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def fig_series_with_forecast(run: dict, label: str) -> go.Figure:
    fig = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
    fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines+markers",
                             name="Actual",
                             line=dict(color=MPA_INK, width=2),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=run["trend"].index, y=run["trend"].values,
                             mode="lines", name="Trend",
                             line=dict(color=MPA_SAND, width=2, dash="dash")))
    # split fitted into in-sample and forecast
    cutoff = run["n_train"]
    full_idx = run["full_index"]
    in_idx = full_idx[:cutoff]
    fc_idx = full_idx[cutoff:]
    fig.add_trace(go.Scatter(x=in_idx, y=run["fitted"].iloc[:cutoff].values,
                             mode="lines", name=f"Fit ({label})",
                             line=dict(color=MPA_PINE, width=2.5)))
    fig.add_trace(go.Scatter(x=fc_idx, y=run["fitted"].iloc[cutoff:].values,
                             mode="lines+markers", name=f"Forecast ({label})",
                             line=dict(color=MPA_PINE, width=2.5, dash="dot"),
                             marker=dict(size=6, symbol="diamond")))
    sigma = run["residuals"].std()
    if not np.isnan(sigma):
        upper = run["fitted"].iloc[cutoff:].values + 1.96 * sigma
        lower = run["fitted"].iloc[cutoff:].values - 1.96 * sigma
        fig.add_trace(go.Scatter(x=list(fc_idx) + list(fc_idx[::-1]),
                                 y=list(upper) + list(lower[::-1]),
                                 fill="toself", fillcolor="rgba(31,77,63,0.12)",
                                 line=dict(color="rgba(0,0,0,0)"),
                                 hoverinfo="skip", showlegend=True,
                                 name="±1.96σ band"))
    fig.update_layout(title=f"Actual vs Fit + Forecast — {label}",
                      height=420, hovermode="x unified")
    return fig


def fig_decomposition(run: dict, label: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Trend", "Detrended (seasonal + noise)",
                                        "Residuals"),
                        vertical_spacing=0.07)
    in_idx = run["trend"].index[: run["n_train"]]
    fig.add_trace(go.Scatter(x=in_idx,
                             y=run["trend"].iloc[: run["n_train"]].values,
                             line=dict(color=MPA_SAND, width=2.2),
                             name="Trend", showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=run["detrended"].index, y=run["detrended"].values,
                             line=dict(color=MPA_SLATE, width=1.8),
                             name="Detrended", showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=run["residuals"].index, y=run["residuals"].values,
                             mode="lines+markers",
                             line=dict(color=MPA_PINE, width=1.4),
                             marker=dict(size=4),
                             name="Residuals", showlegend=False),
                  row=3, col=1)
    fig.add_hline(y=0 if label == "additive" else 1, line_color="#aaa",
                  line_dash="dot", row=2, col=1)
    fig.add_hline(y=0, line_color="#aaa", line_dash="dot", row=3, col=1)
    fig.update_layout(height=520,
                      paper_bgcolor=MPA_PARCHMENT,
                      plot_bgcolor=MPA_PARCHMENT,
                      title=f"Decomposition — {label}",
                      font=dict(family="Inter, system-ui, sans-serif",
                                color=MPA_INK))
    for r in (1, 2, 3):
        fig.update_xaxes(gridcolor="#E6E2D6", row=r, col=1)
        fig.update_yaxes(gridcolor="#E6E2D6", row=r, col=1)
    return fig


def fig_seasonal_indices(run: dict, label: str) -> go.Figure:
    idx = run["seasonal_index"]
    fig = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
    fig.add_trace(go.Bar(x=[f"k={k}" for k in idx.index], y=idx.values,
                         marker_color=MPA_PINE,
                         text=[f"{v:.3f}" for v in idx.values],
                         textposition="outside",
                         textfont=dict(color=MPA_SLATE)))
    base = 0 if label == "additive" else 1
    fig.add_hline(y=base, line_color=MPA_SAND, line_dash="dash",
                  annotation_text=f"neutral = {base}",
                  annotation_position="bottom right")
    fig.update_layout(title=f"Adjusted seasonal indices — {label}",
                      height=320, yaxis_title="Index value",
                      xaxis_title="Season-of-cycle k")
    return fig


# ---------------------------------------------------------------------------
# Layout the body
# ---------------------------------------------------------------------------
for label, run in runs.items():
    st.markdown(f"## {label.title()} model")

    st.plotly_chart(fig_series_with_forecast(run, label), use_container_width=True)

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.plotly_chart(fig_decomposition(run, label), use_container_width=True)
    with c2:
        st.plotly_chart(fig_seasonal_indices(run, label), use_container_width=True)

        st.markdown("##### Adjusted seasonal indices")
        idx_df = pd.DataFrame({"season_k": run["seasonal_index"].index,
                               "index": run["seasonal_index"].values})
        st.dataframe(idx_df.style.format({"index": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

    # metrics tables
    m_in = run["metrics"]
    m_out = run["holdout_metrics"]
    metric_df = pd.DataFrame({
        "Metric": ["MAE / MAD", "MAPE (%)", "RMSE", "n"],
        "In-sample fit": [m_in["MAE"], m_in["MAPE_%"], m_in["RMSE"], m_in["n"]],
        "Hold-out back-test": [m_out["MAE"], m_out["MAPE_%"], m_out["RMSE"], m_out["n"]],
    })
    st.markdown("##### Error metrics")
    st.dataframe(
        metric_df.style.format({"In-sample fit": "{:,.3f}",
                                "Hold-out back-test": "{:,.3f}"}),
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------------
# If user compared both, render a head-to-head ranking
# ---------------------------------------------------------------------------
if len(runs) > 1:
    st.markdown("## Head-to-head ranking")
    rows = []
    for label, run in runs.items():
        rows.append({
            "Model": f"{trend_method_short} × {label}",
            "MAE (fit)": run["metrics"]["MAE"],
            "MAPE % (fit)": run["metrics"]["MAPE_%"],
            "MAE (hold-out)": run["holdout_metrics"]["MAE"],
            "MAPE % (hold-out)": run["holdout_metrics"]["MAPE_%"],
        })
    rank_df = pd.DataFrame(rows).set_index("Model")
    sort_col = "MAPE % (hold-out)" if not rank_df["MAPE % (hold-out)"].isna().all() else "MAPE % (fit)"
    rank_df = rank_df.sort_values(sort_col)
    st.dataframe(
        rank_df.style.format("{:,.3f}").background_gradient(
            cmap="Greens_r", subset=[sort_col], axis=0),
        use_container_width=True,
    )
    st.caption(f"Best model by {sort_col}: **{rank_df.index[0]}** "
               f"({rank_df.iloc[0][sort_col]:.2f}).")

# ---------------------------------------------------------------------------
# Download the forecast
# ---------------------------------------------------------------------------
st.markdown("## Export")
first_run = runs[list(runs.keys())[0]]
# Build the actual column positionally — robust to rebuilt timelines whose
# labels may not exactly match the source y.index
full_idx = first_run["full_index"]
actual_col = np.full(len(full_idx), np.nan, dtype=float)
actual_col[: len(y)] = y.values
export_df = pd.DataFrame({
    "date": full_idx,
    "actual": actual_col,
    "trend": first_run["trend"].values,
    "forecast": first_run["fitted"].values,
})
csv = export_df.to_csv(index=False).encode()
st.download_button(
    label="Download forecast as CSV",
    data=csv,
    file_name=f"forecast_{trend_method_short}_{list(runs.keys())[0]}.csv",
    mime="text/csv",
)

st.markdown(
    f"<hr style='border-top:1px solid #E6E2D6;'>"
    f"<div style='color:{MPA_SUBTLE}; font-size:0.85rem; text-align:center;'>"
    f"The Mountain Path Academy · Financial Risk &amp; Forecasting Lab"
    f"</div>",
    unsafe_allow_html=True,
)
