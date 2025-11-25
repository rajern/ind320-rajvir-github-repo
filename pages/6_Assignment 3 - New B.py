import streamlit as st
import numpy as np
import pandas as pd
from scipy.fft import dct, idct
from sklearn.neighbors import LocalOutlierFactor
import plotly.graph_objects as go

from src.data_loader import load_open_meteo_api

st.title("Assignment 3 – Outliers and anomalies (SPC & LOF)")


# ---- Functions created in assignment3.ipynb ----

# Function for plotting temperature and relevant summaries of outliers
def plot_temperature_with_spc(
    df: pd.DataFrame,
    time_col="date",
    temp_col="temperature_2m",
    trend_keep_fraction=0.02,  # how much of the lowest DCT frequencies to keep for trend
    sigma_threshold=3.0        # sigma threshold for SPC limits
):
    # Ensure chronological order and extract arrays
    df = df.sort_values(time_col).reset_index(drop=True)
    timestamps = pd.to_datetime(df[time_col])
    temp = df[temp_col].to_numpy(dtype=float)

    # Simple NaN handling: interpolate missing values
    if np.isnan(temp).any():
        temp = pd.Series(temp).interpolate(limit_direction="both").to_numpy()

    n_samples = len(temp)

    # --------- DCT: separate low-frequency (trend) and high-frequency (variations) ----------
    coeffs = dct(temp, type=2, norm="ortho")

    # Number of lowest frequencies to keep for the smooth seasonal trend
    keep = max(1, int(trend_keep_fraction * n_samples))

    trend_coeffs = np.zeros_like(coeffs)
    trend_coeffs[:keep] = coeffs[:keep]
    seasonal_trend = idct(trend_coeffs, type=2, norm="ortho")

    # Seasonally Adjusted Temperature Variations (SATV)
    satv = temp - seasonal_trend

    # --------- Robust SPC statistics on SATV ----------
    satv_center = np.median(satv)
    satv_mad = np.median(np.abs(satv - satv_center))
    # Convert MAD to a normal-consistent sigma; fall back to std if MAD==0
    robust_sigma = 1.4826 * satv_mad if satv_mad > 0 else np.std(satv)

    satv_lower = satv_center - sigma_threshold * robust_sigma
    satv_upper = satv_center + sigma_threshold * robust_sigma

    # Map SPC limits back to temperature scale by adding the trend
    lower_limit = seasonal_trend + satv_lower
    upper_limit = seasonal_trend + satv_upper

    # Outliers are points where SATV is outside limits
    is_outlier = (satv < satv_lower) | (satv > satv_upper)

    # --------- Plot using Plotly ----------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=temp,
            mode="lines",
            name="Temperature",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=lower_limit,
            mode="lines",
            name="SPC lower",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=upper_limit,
            mode="lines",
            name="SPC upper",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps[is_outlier],
            y=temp[is_outlier],
            mode="markers",
            name="Outliers",
            marker=dict(color="red", size=5),
        )
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        legend_title=None,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    summary = {
        "n_points": int(n_samples),
        "n_outliers": int(is_outlier.sum()),
        "outlier_fraction": float(is_outlier.mean()),
        "satv_center": float(satv_center),
        "robust_sigma": float(robust_sigma),
        "satv_lower": float(satv_lower),
        "satv_upper": float(satv_upper),
    }

    return fig, summary


# Function for plotting precipitation and relevant summaries of outliers
def plot_precipitation_with_lof(
    df,
    time_col="date",
    precip_col="precipitation",
    outlier_fraction=0.01,  # desired share of outliers (e.g. 0.01 = 1%)
    n_neighbors=20          # neighbors used by LOF
):
    
    # Ensure chronological order and extract arrays
    df = df.sort_values(time_col).reset_index(drop=True)
    time = pd.to_datetime(df[time_col])
    precip = df[precip_col].to_numpy(dtype=float)

    # Simple NaN handling: interpolate missing values
    if np.isnan(precip).any():
        precip = pd.Series(precip).interpolate(limit_direction="both").to_numpy()

    n = len(precip)

    # LOF expects a 2D feature matrix
    X = precip.reshape(-1, 1)

    # Make sure n_neighbors is valid
    n_neighbors = max(5, min(n_neighbors, n - 1))

    # Fit Local Outlier Factor model
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=outlier_fraction,
        novelty=False
    )
    labels = lof.fit_predict(X)  # 1 = inlier, -1 = outlier

    is_outlier = labels == -1

    # --------- Plot using Plotly ----------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=precip,
            mode="lines",
            name="Precipitation",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time[is_outlier],
            y=precip[is_outlier],
            mode="markers",
            name="Outliers",
            marker=dict(color="red", size=5),
        )
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Precipitation (mm)",
        legend_title=None,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    # Simple summary of outliers
    n_outliers = int(is_outlier.sum())
    summary = {
        "n_points": int(n),
        "n_outliers": n_outliers,
        "outlier_fraction_estimated": float(n_outliers / n),
        "precip_min_outlier": float(precip[is_outlier].min()) if n_outliers > 0 else None,
        "precip_max_outlier": float(precip[is_outlier].max()) if n_outliers > 0 else None,
    }

    return fig, summary


# Load Open-Meteo data based on price area
AREA_COORDS = {
    "NO1": (59.91390, 10.75220),  # Oslo
    "NO2": (58.14670, 7.99560),   # Kristiansand
    "NO3": (63.43050, 10.39510),  # Trondheim
    "NO4": (69.64920, 18.95600),  # Tromsø
    "NO5": (60.39299, 5.32415),   # Bergen
}

# Use shared selection from Production explorer
pricearea = st.session_state.get("pricearea", "NO1")
if pricearea not in AREA_COORDS:
    pricearea = "NO1"

lat, lon = AREA_COORDS[pricearea]

st.write(f"Current price area: **{pricearea}**")

# Load hourly weather data for this area and 2021
df = load_open_meteo_api(latitude=lat, longitude=lon, year=2021, area=pricearea)

# Make a copy with an explicit date column to match the notebook functions
df_plot = df.reset_index().rename(columns={"time": "date"})


# ------------------------------------------------------------
# Tabs: SPC (temperature) and LOF (precipitation)
# ------------------------------------------------------------

tab_spc, tab_lof = st.tabs(["SPC – temperature", "LOF – precipitation"])

# ---------------- SPC tab ----------------
with tab_spc:
    st.subheader("DCT + SPC on temperature")

    c1, c2 = st.columns(2)
    trend_keep_fraction = c1.number_input(
        "Trend keep fraction",
        min_value=0.001,
        max_value=0.5,
        value=0.02,
        step=0.005,
    )
    sigma_threshold = c2.number_input(
        "Sigma threshold",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.5,
    )

    fig_spc, summary_spc = plot_temperature_with_spc(
        df_plot,
        time_col="date",
        temp_col="temperature_2m",
        trend_keep_fraction=trend_keep_fraction,
        sigma_threshold=sigma_threshold,
    )
    st.plotly_chart(fig_spc, use_container_width=True)
    st.json(summary_spc)

# ---------------- LOF tab ----------------
with tab_lof:
    st.subheader("LOF on precipitation")

    c1, c2 = st.columns(2)
    outlier_fraction = c1.number_input(
        "Desired outlier fraction",
        min_value=0.001,
        max_value=0.2,
        value=0.01,
        step=0.005,
    )
    n_neighbors = c2.number_input(
        "Number of neighbors",
        min_value=5,
        max_value=100,
        value=20,
        step=1,
    )

    fig_lof, summary_lof = plot_precipitation_with_lof(
        df_plot,
        time_col="date",
        precip_col="precipitation",
        outlier_fraction=outlier_fraction,
        n_neighbors=int(n_neighbors),
    )
    st.plotly_chart(fig_lof, use_container_width=True)
    st.json(summary_lof)
