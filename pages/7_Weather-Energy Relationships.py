import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis_context import context_caption, render_analysis_context
from src.data_loader import (
    load_open_meteo_api,
    load_elhub_production_data,
    load_elhub_consumption_data,
)

# ------------------------------------------------------------------
# Configuration and helpers
# ------------------------------------------------------------------

METEO_OPTIONS = {
    "Temperature 2m [\u00b0C]": "temperature_2m",
    "Precipitation [mm]": "precipitation",
    "Wind speed 10m [m/s]": "wind_speed_10m",
    "Wind gusts 10m [m/s]": "wind_gusts_10m",
}

# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------

context = render_analysis_context()

st.title("Weather-energy relationships")
st.caption(
    "Compare weather with electricity production or consumption using a "
    "sliding-window correlation. Positive lag means weather leads the energy series."
)
st.markdown(context_caption(context))
st.caption(f"Coordinates: {context.latitude:.4f}, {context.longitude:.4f}")

area = context.price_area
year = context.start_date.year

st.divider()
st.subheader("Analysis settings")
settings_container = st.container()
results_container = st.container()

with settings_container:
    meteo_label = st.selectbox(
        "Meteorological property",
        list(METEO_OPTIONS.keys()),
    )
    meteo_col = METEO_OPTIONS[meteo_label]

    # Load Elhub data (cached inside data_loader)
    df_prod = load_elhub_production_data()
    df_cons = load_elhub_consumption_data()

    # Build available groups dynamically for the selected price area
    prod_groups = sorted(
        df_prod.loc[df_prod["pricearea"] == area, "productiongroup"].dropna().unique()
    )
    cons_groups = sorted(
        df_cons.loc[df_cons["pricearea"] == area, "consumptiongroup"].dropna().unique()
    )

    energy_mode = st.radio(
        "Energy type",
        ["Production", "Consumption"],
        index=0,
        help="Choose whether to correlate production or consumption with weather.",
    )

    if energy_mode == "Production":
        group_options = prod_groups
        group_label = "Production group"
    else:
        group_options = cons_groups
        group_label = "Consumption group"

    if not group_options:
        st.error(f"No {energy_mode.lower()} groups found for {area}.")
        st.stop()

    energy_group = st.selectbox(group_label, group_options)

    # Window and lag controls
    window_days = st.slider(
        "Window length (days)",
        min_value=3,
        max_value=60,
        value=14,
        step=1,
        help="Length of the sliding window used for correlation.",
    )
    window_hours = window_days * 24

    lag_hours = st.slider(
        "Lag (hours, weather -> energy)",
        min_value=-72,
        max_value=72,
        value=0,
        step=1,
        help=(
            "Positive lag: energy responds later than weather. "
            "Negative lag: energy leads weather."
        ),
    )

    display_resolution = context.resolution
    st.caption(
        f"Charts use the shared {display_resolution.lower()} resolution. Correlation "
        "is calculated on aligned hourly observations."
    )
    run_analysis = st.button("Run relationship analysis", type="primary")

if not run_analysis:
    results_container.info(
        "Review the settings and click **Run relationship analysis** to calculate the results."
    )
    st.stop()

# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

latitude = context.latitude
longitude = context.longitude

# Weather from Open-Meteo API (cached in data_loader)
meteo_df = load_open_meteo_api(
    latitude=latitude,
    longitude=longitude,
    year=year,
    area=area,
)

if meteo_col not in meteo_df.columns:
    results_container.error(f"Column '{meteo_col}' not found in Open-Meteo data.")
    st.stop()

meteo_series = (
    meteo_df[[meteo_col]]
    .rename(columns={meteo_col: "meteo"})
    .sort_index()
)
# Ensure unique index
meteo_series = meteo_series[~meteo_series.index.duplicated(keep="first")]

# Energy series from MongoDB (already cached by data_loader)
if energy_mode == "Production":
    df_energy = df_prod[
        (df_prod["pricearea"] == area)
        & (df_prod["productiongroup"] == energy_group)
        & (df_prod["starttime"].dt.year == year)
    ].copy()
    energy_series_name = f"Production | {energy_group}"
else:
    df_energy = df_cons[
        (df_cons["pricearea"] == area)
        & (df_cons["consumptiongroup"] == energy_group)
        & (df_cons["starttime"].dt.year == year)
    ].copy()
    energy_series_name = f"Consumption | {energy_group}"

if df_energy.empty:
    results_container.warning(
        f"No {energy_mode.lower()} data for group '{energy_group}' in {area} for {year}."
    )
    st.stop()

df_energy["starttime"] = pd.to_datetime(df_energy["starttime"])

# Aggregate to one value per timestamp and ensure unique index
energy_series = (
    df_energy.groupby("starttime")["quantitykwh"]
    .sum()
    .sort_index()
    .rename("energy")
)
energy_series = energy_series[~energy_series.index.duplicated(keep="first")]

# Align to common hourly time index (inner join)
combined = pd.concat([meteo_series, energy_series], axis=1, join="inner").dropna()

start_ts = pd.Timestamp(context.start_date)
end_ts = pd.Timestamp(context.end_date) + pd.Timedelta(days=1)
combined = combined[(combined.index >= start_ts) & (combined.index < end_ts)]

if combined.empty:
    results_container.warning("No overlapping timestamps between weather and energy data.")
    st.stop()

# Apply lag: positive lag => energy responds later than weather
if lag_hours != 0:
    combined["energy_lagged"] = combined["energy"].shift(-lag_hours)
else:
    combined["energy_lagged"] = combined["energy"]

combined = combined.dropna(subset=["meteo", "energy_lagged"])

if len(combined) < window_hours:
    results_container.warning(
        "Window length is longer than the available time series after lagging. "
        "Reduce the window length or lag."
    )
    st.stop()

# Sliding window correlation (centered window)
corr_series = (
    combined["meteo"]
    .rolling(window=window_hours, min_periods=max(10, window_hours // 5), center=True)
    .corr(combined["energy_lagged"])
)

# Overall Pearson correlation for reference
overall_corr = combined["meteo"].corr(combined["energy_lagged"])


if display_resolution == "Daily":
    plot_combined = combined.resample("D").mean()
elif display_resolution == "Monthly":
    plot_combined = combined.resample("MS").mean()
else:
    plot_combined = combined
# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

with results_container:
    st.subheader("Results")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Overall correlation", f"{overall_corr:.3f}")
    metric_2.metric("Applied lag", f"{lag_hours} h")
    metric_3.metric("Hourly observations", f"{len(combined):,}")

    st.subheader("Weather and energy over time")
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

    fig_ts.add_trace(
        go.Scatter(
            x=plot_combined.index,
            y=plot_combined["meteo"],
            name=meteo_label,
            mode="lines",
        ),
        secondary_y=False,
    )

    fig_ts.add_trace(
        go.Scatter(
            x=plot_combined.index,
            y=plot_combined["energy_lagged"] / 1_000_000,
            name=f"{energy_series_name} (lagged)",
            mode="lines",
        ),
        secondary_y=True,
    )

    fig_ts.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_ts.update_xaxes(title_text="Time")
    fig_ts.update_yaxes(title_text=meteo_label, secondary_y=False)
    fig_ts.update_yaxes(title_text="Average quantity (GWh)", secondary_y=True)

    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Sliding-window correlation")

    fig_corr = go.Figure()
    fig_corr.add_trace(
        go.Scatter(
            x=corr_series.index,
            y=corr_series,
            mode="lines",
            name="Rolling correlation",
        )
    )
    fig_corr.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis=dict(title="Correlation", range=[-1, 1]),
        xaxis=dict(title="Time"),
        shapes=[
            # zero line
            dict(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                y0=0,
                y1=0,
                line=dict(width=1, dash="dot"),
            )
        ],
    )

    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Correlation ranges from -1 to 1. Values near 1 move together, values near "
        "-1 move in opposite directions, and values near 0 show little linear relationship."
    )

    st.markdown(
        f"""
**Summary**

- Overall Pearson correlation (selected period, with lag {lag_hours} h): `{overall_corr:.3f}`
- Rolling window: `{window_days}` days (`{window_hours}` hours)
- Meteo series: **{meteo_label}**
- Energy series: **{energy_series_name}** in **{area}**
"""
    )
