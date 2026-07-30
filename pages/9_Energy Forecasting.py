import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

st.title("Energy forecasting")
st.caption(
    "Fit a configurable SARIMAX model to hourly or daily energy data and compare fitted values, "
    "forecasts and confidence intervals."
)

# Load data once (cached in data_loader)
prod_df = load_elhub_production_data()
cons_df = load_elhub_consumption_data()

# Layout: controls on the left, plot on the right
left_col, right_col = st.columns([1, 2])

# ---------- Controls (left) ---------- #
with left_col:
    st.header("Controls")

    # Dataset selection
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

    # Price areas
    areas = sorted(df["pricearea"].dropna().unique().tolist())
    if not areas:
        st.error("No price areas available in the selected dataset.")
        st.stop()

    if "pricearea" not in st.session_state:
        st.session_state["pricearea"] = areas[0]

    default_index = (
        areas.index(st.session_state["pricearea"])
        if st.session_state["pricearea"] in areas
        else 0
    )

    area = st.radio(
        "Price area",
        areas,
        index=default_index,
        horizontal=True,
    )
    st.session_state["pricearea"] = area

    # Groups
    groups = sorted(df[group_col].dropna().unique().tolist())
    group = st.selectbox("Group", groups)

# Build base hourly series
target_hourly = get_target_series(df, area=area, group_col=group_col, group=group)

if target_hourly.empty:
    with right_col:
        st.header("Forecast")
        st.warning("No data found for this combination of dataset, price area and group.")
    st.stop()

with left_col:
    # Frequency selection
    st.subheader("Frequency and training period")

    freq_label = st.selectbox("Frequency", ["Hourly", "Daily"], index=0)
    freq = "h" if freq_label == "Hourly" else "D"

    min_date = target_hourly.index.min().date()
    max_date = target_hourly.index.max().date()

    start_date = st.date_input(
        "Start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        st.stop()

    # Slice hourly series on chosen dates
    sliced = target_hourly.loc[str(start_date) : str(end_date)]

    if sliced.empty:
        st.error("No data in the selected training period.")
        st.stop()

    # Convert to chosen frequency
    if freq == "h":
        series = sliced.asfreq("h")
        ylabel = "Quantity (kWh)"

        # Allow up to ~1 year of hourly data (som i praksis er det du trenger)
        max_points = 24 * 365  # ~1 year
        if len(series) > max_points:
            series = series.iloc[-max_points:]
            st.info(
                f"Training data truncated to the last {max_points} hours (~1 year) "
                "for performance reasons."
            )
    else:  # Daily
        series = sliced.resample("D").sum()
        ylabel = "Quantity (kWh/day)"

        # Allow up to ~3 years of daily data
        max_points = 365 * 3
        if len(series) > max_points:
            series = series.iloc[-max_points:]
            st.info(
                f"Training data truncated to the last {max_points} days (~3 years) "
                "for performance reasons."
            )


    train_series = series.astype("float64")

    if len(train_series) < 30:
        st.warning("Training period is very short. Consider selecting a longer period.")

    train_start_actual = train_series.index.min().date()
    train_end_actual = train_series.index.max().date()

    # SARIMAX parameters
    st.subheader("SARIMAX parameters")

    # Choose more informative defaults depending on frequency:
    # - d=0 (ingen vanlig differanse)
    # - sesongdifferanse + sesong-MA for å fange døgn/uke-mønster
    if freq == "h":
        default_p, default_d, default_q = 1, 0, 1
        default_P, default_D, default_Q, default_s = 0, 1, 1, 24   # daglig sesong i timesdata
    else:  # Daily
        default_p, default_d, default_q = 1, 0, 1
        default_P, default_D, default_Q, default_s = 0, 1, 1, 7    # ukesesong i dagsdata

    col_pq1, col_pq2, col_pq3 = st.columns(3)
    with col_pq1:
        p = st.number_input("AR order (p)", min_value=0, max_value=5, value=default_p, step=1)
    with col_pq2:
        d = st.number_input("Diff order (d)", min_value=0, max_value=2, value=default_d, step=1)
    with col_pq3:
        q = st.number_input("MA order (q)", min_value=0, max_value=5, value=default_q, step=1)

    col_PDQ1, col_PDQ2, col_PDQ3, col_PDQ4 = st.columns(4)
    with col_PDQ1:
        P = st.number_input("Seasonal AR (P)", min_value=0, max_value=5, value=default_P, step=1)
    with col_PDQ2:
        D = st.number_input("Seasonal diff (D)", min_value=0, max_value=2, value=default_D, step=1)
    with col_PDQ3:
        Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=5, value=default_Q, step=1)
    with col_PDQ4:
        s_help = "Seasonal period (e.g. 24 for daily pattern in hourly data, 7 for weekly pattern in daily data)."
        seasonal_period = st.number_input(
            "Seasonal period (s)",
            min_value=1,
            max_value=24 * 14,
            value=default_s,
            step=1,
            help=s_help,
        )


    # Forecast horizon
    st.subheader("Forecast horizon")
    default_h = 24 * 7 if freq == "h" else 30
    max_h = 24 * 60 if freq == "h" else 365
    horizon = st.number_input(
        "Steps ahead",
        min_value=1,
        max_value=max_h,
        value=default_h,
        step=1,
    )

    run_forecast = st.button("Run forecast")

# ---------- Forecast (right) ---------- #
with right_col:
    st.header("Forecast")

    if not run_forecast:
        st.info("Adjust the controls on the left and click **Run forecast** to compute a forecast.")
    else:
        if train_series.empty:
            st.error("No data in the selected training period.")
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
            except Exception as e:
                st.error(f"Model fitting failed: {e}")
            else:
                # In-sample fitted values
                try:
                    fitted = results.fittedvalues
                except Exception:
                    fitted = None

                # Forecast
                forecast_res = results.get_forecast(steps=horizon)
                forecast = forecast_res.predicted_mean
                conf_int = forecast_res.conf_int()

                title = (
                    f"{dataset_type} forecast – {area}, {group} "
                    f"({train_start_actual} to {train_end_actual}, "
                    f"horizon {horizon} {freq_label.lower()} steps)"
                )

                fig = build_forecast_figure(
                    train_series=train_series,
                    fitted=fitted,
                    forecast=forecast,
                    conf_int=conf_int,
                    title=title,
                    ylabel=ylabel,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Simple RMSE on training residuals
                residuals = results.resid
                rmse = float(np.sqrt(np.mean(residuals**2)))
                st.markdown(f"**Training RMSE:** {rmse:,.2f} kWh")
