import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis_context import context_caption, render_analysis_context
from src.data_loader import load_open_meteo_api

context = render_analysis_context()

st.title("Weather explorer")
st.caption(
    "Explore one weather variable at a time for the active location and period."
)
st.markdown(context_caption(context))

WEATHER_VARIABLES = {
    "Temperature": ("temperature_2m", "\u00b0C"),
    "Precipitation": ("precipitation", "mm"),
    "Wind speed": ("wind_speed_10m", "m/s"),
    "Wind direction": ("wind_direction_10m", "\u00b0"),
    "Wind gusts": ("wind_gusts_10m", "m/s"),
}


def circular_mean_degrees(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return float("nan")
    radians = np.deg2rad(values)
    angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.rad2deg(angle) % 360)

pricearea = context.price_area
lat = context.latitude
lon = context.longitude
location_name = context.location_label
df = load_open_meteo_api(
    latitude=lat,
    longitude=lon,
    year=2021,
    area=pricearea,
)

start_ts = pd.Timestamp(context.start_date)
end_ts = pd.Timestamp(context.end_date) + pd.Timedelta(days=1)
df_period = df[(df.index >= start_ts) & (df.index < end_ts)]

area_col, location_col, period_col = st.columns(3)
area_col.metric("Price area", pricearea)
location_col.metric("Weather location", location_name)
period_col.metric(
    "Period",
    f"{context.start_date:%d %b} - {context.end_date:%d %b %Y}",
)
st.caption(f"Coordinates: {lat:.4f}, {lon:.4f}")

st.divider()
explore_tab, overview_tab = st.tabs(["Explore", "Data overview"])

with explore_tab:
    st.subheader("Analysis settings")
    control_col, range_col = st.columns([1, 2])
    with control_col:
        variable_label = st.selectbox(
            "Weather variable",
            list(WEATHER_VARIABLES),
            index=0,
        )
        variable, unit = WEATHER_VARIABLES[variable_label]

    with range_col:
        st.info(
            f"Time resolution: **{context.resolution}**. "
            "Change the shared period and resolution in the sidebar."
        )

    if context.resolution == "Hourly":
        df_subset = df_period[[variable]].copy()
    else:
        resolution_rule = "D" if context.resolution == "Daily" else "MS"
        if variable == "precipitation":
            df_subset = df_period[[variable]].resample(resolution_rule).sum()
        elif variable == "wind_direction_10m":
            df_subset = (
                df_period[[variable]]
                .resample(resolution_rule)
                .agg(circular_mean_degrees)
            )
        else:
            df_subset = df_period[[variable]].resample(resolution_rule).mean()
        df_subset = df_subset.dropna()

    st.subheader("Results")
    summary_1, summary_2, summary_3 = st.columns(3)
    summary_1.metric("Average", f"{df_subset[variable].mean():,.1f} {unit}")
    summary_2.metric("Minimum", f"{df_subset[variable].min():,.1f} {unit}")
    summary_3.metric("Maximum", f"{df_subset[variable].max():,.1f} {unit}")

    fig = px.line(
        df_subset.reset_index(),
        x="time",
        y=variable,
        labels={"time": "Time", variable: f"{variable_label} ({unit})"},
        title=(
            f"{variable_label} | {location_name}, "
            f"{context.start_date:%d %b} to {context.end_date:%d %b %Y}"
        ),
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title=f"{variable_label} ({unit})",
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The summary metrics and chart use the active weather location, period, "
        "and time resolution."
    )

with overview_tab:
    st.subheader("Variables in the selected period")
    overview_rows = []
    for label, (column, unit) in WEATHER_VARIABLES.items():
        overview_rows.append(
            {
                "Variable": label,
                "Unit": unit,
                "Mean": df_period[column].mean(),
                "Minimum": df_period[column].min(),
                "Maximum": df_period[column].max(),
            }
        )

    overview = pd.DataFrame(overview_rows)
    st.dataframe(
        overview,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Mean": st.column_config.NumberColumn(format="%.1f"),
            "Minimum": st.column_config.NumberColumn(format="%.1f"),
            "Maximum": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "Hourly ERA5 weather data from Open-Meteo. The shared sidebar controls "
        "the location and period used here."
    )
