import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.analysis_context import render_analysis_context
from src.data_loader import (
    load_elhub_production_data,
    load_elhub_consumption_data,
)


# ---------- Helper functions ---------- #

def get_target_series(df: pd.DataFrame, area: str, group_col: str, group: str) -> pd.Series:
    """
    Filter Elhub data for one price area and one group, return a clean hourly series.

    - Aggregates any duplicate timestamps by summing per hour.
    - Returns a Series with hourly ('h') frequency and float values.
    """
    sub = df[(df["pricearea"] == area) & (df[group_col] == group)].copy()

    if sub.empty:
        return pd.Series(dtype="float64")

    sub["starttime"] = pd.to_datetime(sub["starttime"])
    sub = sub.sort_values("starttime")

    # Aggregate per hour to avoid duplicate timestamps
    series = (
        sub.set_index("starttime")["quantitykwh"]
        .astype("float64")
        .resample("h")          # hourly grid (lowercase 'h' to avoid warning)
        .sum(min_count=1)
    )

    # Fill gaps by interpolation
    series = series.interpolate(limit_direction="both")

    return series


def build_forecast_figure(
    train_series: pd.Series,
    fitted: pd.Series | None,
    forecast: pd.Series,
    conf_int: pd.DataFrame,
    title: str,
    ylabel: str,
) -> go.Figure:
    """
    Create a Plotly figure with training data, fitted values, forecast and confidence interval.
    """
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    fig = go.Figure()

    # Training data
    fig.add_trace(
        go.Scatter(
            x=train_series.index,
            y=train_series.values,
            mode="lines",
            name="Training data",
        )
    )

    # In-sample fitted values (if available)
    if fitted is not None:
        fig.add_trace(
            go.Scatter(
                x=fitted.index,
                y=fitted.values,
                mode="lines",
                name="Fitted (in-sample)",
                line=dict(width=1, dash="dot"),
            )
        )

    # Confidence interval band
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=upper.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=lower.values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Confidence interval",
            hoverinfo="skip",
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines",
            name="Forecast",
        )
    )

    # Vertical line at training end
    fig.add_vline(
        x=train_series.index[-1],
        line_width=1,
        line_dash="dot",
        line_color="gray",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=ylabel,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


# ---------- Streamlit page ---------- #

context = render_analysis_context(
    show_period=False,
    show_resolution=False,
    show_location=False,
)

st.title("Energy forecasting")
st.caption(
    "Fit a configurable SARIMAX model to hourly or daily energy data and compare fitted values, "
    "forecasts and confidence intervals."
)
st.markdown(f"**Active selection:** {context.price_area}")

# Load data once (cached in data_loader)
prod_df = load_elhub_production_data()
cons_df = load_elhub_consumption_data()

st.divider()
st.subheader("Analysis settings")
settings_container = st.container()
results_container = st.container()

with settings_container:
    dataset_type = st.radio(
        "Dataset",
        ["Production", "Consumption"],
        horizontal=True,
    )

    if dataset_type == "Production":
        df = prod_df.copy()
        group_col = "productiongroup"
    else:
        df = cons_df.copy()
        group_col = "consumptiongroup"

    areas = sorted(df["pricearea"].dropna().unique().tolist())
    if not areas:
        st.error("No price areas are available in the selected dataset.")
        st.stop()

    area = context.price_area
    if area not in areas:
        st.error(f"No data are available for the active price area {area}.")
        st.stop()

    groups = sorted(
        df.loc[df["pricearea"] == area, group_col].dropna().unique().tolist()
    )
    if not groups:
        st.error("No groups are available for this price area.")
        st.stop()
    group = st.selectbox("Group", groups)

    target_hourly = get_target_series(df, area=area, group_col=group_col, group=group)
    if target_hourly.empty:
        st.error("No data were found for this selection.")
        st.stop()

    freq_label = st.selectbox("Frequency", ["Daily", "Hourly"], index=0)
    freq = "D" if freq_label == "Daily" else "h"

    default_horizon = 30 if freq == "D" else 24 * 7
    max_horizon = 365 if freq == "D" else 24 * 60
    horizon = st.number_input(
        "Forecast horizon",
        min_value=1,
        max_value=max_horizon,
        value=default_horizon,
        step=1,
        help=f"Number of {freq_label.lower()} steps to forecast.",
    )

    min_date = target_hourly.index.min().date()
    max_date = target_hourly.index.max().date()

    if freq == "h":
        default_p, default_d, default_q = 1, 0, 1
        default_P, default_D, default_Q, default_s = 0, 1, 1, 24
    else:
        default_p, default_d, default_q = 1, 0, 1
        default_P, default_D, default_Q, default_s = 0, 1, 1, 7

    with st.expander("Advanced settings"):
        start_date = st.date_input(
            "Training start",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
        end_date = st.date_input(
            "Training end",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

        st.markdown("**SARIMAX order**")
        col_p, col_d, col_q = st.columns(3)
        p = col_p.number_input("p", 0, 5, default_p, 1)
        d = col_d.number_input("d", 0, 2, default_d, 1)
        q = col_q.number_input("q", 0, 5, default_q, 1)

        st.markdown("**Seasonal order**")
        col_P, col_D, col_Q, col_s = st.columns(4)
        P = col_P.number_input("P", 0, 5, default_P, 1)
        D = col_D.number_input("D", 0, 2, default_D, 1)
        Q = col_Q.number_input("Q", 0, 5, default_Q, 1)
        seasonal_period = col_s.number_input(
            "s",
            min_value=1,
            max_value=24 * 14,
            value=default_s,
            step=1,
            help="24 represents a daily pattern in hourly data; 7 represents a weekly pattern in daily data.",
        )

    if start_date > end_date:
        st.error("Training start must be before or equal to training end.")
        st.stop()

    sliced = target_hourly.loc[str(start_date):str(end_date)]
    if freq == "h":
        series = sliced.asfreq("h")
        max_points = 24 * 365
        period_unit = "hour"
        chart_unit = "GWh/hour"
    else:
        series = sliced.resample("D").sum()
        max_points = 365 * 3
        period_unit = "day"
        chart_unit = "GWh/day"

    if len(series) > max_points:
        series = series.iloc[-max_points:]
        st.info(f"Training data use the latest {max_points:,} observations for performance.")

    train_series = series.astype("float64")
    if len(train_series) < 30:
        st.warning("The training period is short; forecast uncertainty may be high.")

    run_forecast = st.button("Run forecast", type="primary")

with results_container:
    st.subheader("Results")

    if not run_forecast:
        st.info(
            "Daily frequency is the practical default. Review the selection and click "
            "**Run forecast**; model parameters are available under Advanced settings."
        )
    elif train_series.empty:
        st.error("No data are available in the selected training period.")
    else:
        try:
            with st.spinner("Fitting SARIMAX model..."):
                model = SARIMAX(
                    train_series,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    freq=freq,
                )
                results = model.fit(disp=False, maxiter=50)
        except Exception as exc:
            st.error(f"Model fitting failed: {exc}")
        else:
            try:
                fitted = results.fittedvalues
            except Exception:
                fitted = None

            forecast_res = results.get_forecast(steps=horizon)
            forecast = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()
            residuals = results.resid
            rmse_kwh = float(np.sqrt(np.mean(residuals**2)))

            metric_1, metric_2, metric_3 = st.columns(3)
            metric_1.metric("Training observations", f"{len(train_series):,}")
            metric_2.metric("Forecast horizon", f"{horizon} {period_unit}s")
            metric_3.metric("Training RMSE", f"{rmse_kwh / 1_000_000:,.2f} GWh")

            title = (
                f"{dataset_type} forecast | {area}, {group} | "
                f"{train_series.index.min().date()} to {train_series.index.max().date()}"
            )
            fig = build_forecast_figure(
                train_series=train_series / 1_000_000,
                fitted=fitted / 1_000_000 if fitted is not None else None,
                forecast=forecast / 1_000_000,
                conf_int=conf_int / 1_000_000,
                title=title,
                ylabel=f"Quantity ({chart_unit})",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The shaded interval represents model uncertainty. This is a portfolio "
                "demonstration, not an operational forecast."
            )
