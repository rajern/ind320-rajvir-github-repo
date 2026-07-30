from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.analysis_context import render_analysis_context
from src.data_loader import load_open_meteo_api
from src.Snow_drift import (
    compute_yearly_results,
    compute_average_sector,
)


def download_weather_for_seasons(
    latitude: float,
    longitude: float,
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    """
    Download hourly weather data from Open-Meteo for all calendar years
    needed to cover the requested seasons.

    A season is defined as:
        1 July of year s  ->  30 June of year s+1.

    To cover seasons [start_season, end_season], we need calendar years:
        start_season ... end_season + 1.
    """
    years: List[int] = list(range(start_season, end_season + 2))

    frames: List[pd.DataFrame] = []

    for year in years:
        # load_open_meteo_api is cached, so repeated calls for the same
        # year and coordinates will be fast.
        df_year = load_open_meteo_api(latitude=latitude, longitude=longitude, year=year)
        frames.append(df_year)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames).sort_index()

    # Remove any duplicate timestamps if they occur
    df_all = df_all[~df_all.index.duplicated(keep="first")]

    return df_all


def prepare_snowdrift_dataframe(
    df_weather: pd.DataFrame,
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    """
    Prepare a DataFrame compatible with the functions in Snow_drift.py.

    - Move the DatetimeIndex to a 'time' column.
    - Rename columns to the names expected by Snow_drift.py.
    - Create a 'season' column:
        season = year if month >= 7, otherwise year - 1
    - Filter to seasons in [start_season, end_season].
    """
    if df_weather.empty:
        return pd.DataFrame()

    df = df_weather.copy().reset_index()  # index -> 'time'
    df.rename(
        columns={
            "time": "time",
            "temperature_2m": "temperature_2m (?C)",
            "precipitation": "precipitation (mm)",
            "wind_speed_10m": "wind_speed_10m (m/s)",
            "wind_direction_10m": "wind_direction_10m (?)",
        },
        inplace=True,
    )

    # Create season label: July?December belong to current year,
    # January?June belong to previous year.
    df["season"] = df["time"].apply(
        lambda dt: dt.year if dt.month >= 7 else dt.year - 1
    )

    # Keep only the requested seasons
    mask = (df["season"] >= start_season) & (df["season"] <= end_season)
    df = df.loc[mask].reset_index(drop=True)

    return df


def plot_yearly_snow_transport(yearly_df: pd.DataFrame):
    """
    Plot yearly mean snow transport Qt in tonnes/m using Plotly.
    """
    df_plot = yearly_df.copy()
    df_plot["Qt (tonnes/m)"] = df_plot["Qt (kg/m)"] / 1000.0

    fig = px.bar(
        df_plot,
        x="season",
        y="Qt (tonnes/m)",
        labels={
            "season": "Season (start year)",
            "Qt (tonnes/m)": "Qt (tonnes/m)",
        },
        title="Yearly mean snow transport per season",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_wind_rose(avg_sector_values: np.ndarray, overall_avg_kgm: float):
    """
    Create a polar wind-rose plot of average directional snow transport
    using Plotly.

    Parameters
    ----------
    avg_sector_values : array-like of length 16
        Average transport for each wind sector in kg/m.
    overall_avg_kgm : float
        Overall average yearly snow transport Qt in kg/m.
    """
    num_sectors = len(avg_sector_values)
    if num_sectors == 0:
        return go.Figure()

    # Convert to tonnes/m for plotting
    values_tonnes = np.array(avg_sector_values) / 1000.0

    # Angles (degrees) and direction labels
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    theta_deg = np.linspace(0, 360, num_sectors, endpoint=False)

    fig = go.Figure(
        data=go.Barpolar(
            r=values_tonnes,
            theta=theta_deg,
            text=directions,
            hovertemplate="Direction: %{text}<br>Qt: %{r:.3f} tonnes/m<extra></extra>",
        )
    )

    overall_tonnes = overall_avg_kgm / 1000.0

    fig.update_layout(
        title=(
            f"Average directional snow transport<br>"
            f"Overall Qt = {overall_tonnes:,.1f} tonnes/m"
        ),
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=theta_deg,
                ticktext=directions,
                direction="clockwise",
                rotation=90,  # 0? = North
            ),
            radialaxis=dict(
                angle=90,
                tickangle=90,
                showline=True,
                linewidth=1,
            ),
        ),
        showlegend=False,
    )

    return fig


# -----------------------------
# Streamlit page
# -----------------------------


def main() -> None:
    context = render_analysis_context(
        show_period=False,
        show_resolution=False,
    )

    st.title("Snow drift")

    st.caption(
        "This page calculates yearly snow drift for a selected coordinate using "
        "hourly weather data from the Open-Meteo archive. "
    )

    lat = context.latitude
    lon = context.longitude
    st.info(
        f"Weather location: {context.location_label} | "
        f"{lat:.4f}, {lon:.4f}"
    )
    st.page_link("pages/5_Energy Map.py", label="Change weather location on Energy Map")

    # --- User controls: seasons and model parameters ---
    st.subheader("Season and model settings")
    st.caption(
        "A year is defined as 1 July in the selected start year to 30 June in the "
        "following year."
    )

    # Season range (stacked, not side by side)
    MIN_SEASON = 2019
    MAX_SEASON = 2024
    start_season, end_season = st.slider(
        "Season range (start year of season)",
        min_value=MIN_SEASON,
        max_value=MAX_SEASON,
        value=(2021, 2023),
    )

    # Model parameters below the slider
    with st.expander("Snow transport model parameters", expanded=False):
        T = st.number_input(
            "Maximum transport distance T (m)",
            min_value=500.0,
            max_value=10000.0,
            value=3000.0,
            step=100.0,
        )
        F = st.number_input(
            "Fetch distance F (m)",
            min_value=1000.0,
            max_value=100000.0,
            value=30000.0,
            step=1000.0,
        )
        theta = st.slider(
            "Relocation coefficient ?",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

    st.write(
        f"Seasons included: **{start_season}?{start_season + 1}** "
        f"to **{end_season}?{end_season + 1}**."
    )

    compute_btn = st.button("Compute snow drift")
    if not compute_btn:
        st.stop()

    # --- Download and prepare weather data ---
    with st.spinner("Downloading hourly weather data from Open-Meteo ..."):
        try:
            df_weather = download_weather_for_seasons(
                latitude=lat,
                longitude=lon,
                start_season=start_season,
                end_season=end_season,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to download weather data from Open-Meteo: {exc}")
            st.stop()

    if df_weather.empty:
        st.error("No weather data returned for this coordinate and season range.")
        st.stop()

    df_snow = prepare_snowdrift_dataframe(
        df_weather=df_weather,
        start_season=start_season,
        end_season=end_season,
    )

    if df_snow.empty:
        st.warning(
            "After preparing the data, no seasons remain in the selected range. "
            "Try widening the season interval."
        )
        st.stop()

    # --- Compute yearly snow transport using functions from Snow_drift.py ---
    yearly_df = compute_yearly_results(df_snow, T=T, F=F, theta=theta)

    if yearly_df.empty:
        st.warning("Snow-drift calculation returned no yearly results.")
        st.stop()

    yearly_df_display = yearly_df.copy()
    yearly_df_display["Snow transport (tonnes/m)"] = yearly_df_display["Qt (kg/m)"] / 1000.0
    yearly_df_display = yearly_df_display.rename(
        columns={
            "season": "Season",
            "Control": "Limiting factor",
            "Qupot (kg/m)": "Wind potential (kg/m)",
            "Qspot (kg/m)": "Snowfall potential (kg/m)",
        }
    )

    overall_avg = float(yearly_df["Qt (kg/m)"].mean())
    metric_1, metric_2 = st.columns(2)
    metric_1.metric("Average seasonal transport", f"{overall_avg / 1000:,.1f} tonnes/m")
    metric_2.metric("Seasons analysed", f"{len(yearly_df):,}")

    st.subheader("Yearly snow transport per season")
    st.dataframe(
        yearly_df_display[
            [
                "Season",
                "Snow transport (tonnes/m)",
                "Limiting factor",
                "Wind potential (kg/m)",
                "Snowfall potential (kg/m)",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Snow transport (tonnes/m)": st.column_config.NumberColumn(format="%.1f"),
            "Wind potential (kg/m)": st.column_config.NumberColumn(format="%.0f"),
            "Snowfall potential (kg/m)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    # --- Plot yearly Qt with Plotly ---
    fig_yearly = plot_yearly_snow_transport(yearly_df)
    st.plotly_chart(fig_yearly, use_container_width=True)

    # --- Wind rose with Plotly ---
    st.subheader("Wind rose for snow transport")

    avg_sectors = compute_average_sector(df_snow)
    overall_avg = float(yearly_df["Qt (kg/m)"].mean())

    fig_rose = plot_wind_rose(avg_sectors, overall_avg_kgm=overall_avg)
    st.plotly_chart(fig_rose, use_container_width=True)


if __name__ == "__main__":
    main()
