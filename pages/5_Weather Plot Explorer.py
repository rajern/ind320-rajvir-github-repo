import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_open_meteo_api

st.title("Weather plot explorer for 2021")

# Map price areas to city coordinates
AREA_COORDS = {
    "NO1": (59.91390, 10.75220),  # Oslo
    "NO2": (58.14670, 7.99560),   # Kristiansand
    "NO3": (63.43050, 10.39510),  # Trondheim
    "NO4": (69.64920, 18.95600),  # Tromsø
    "NO5": (60.39299, 5.32415),   # Bergen
}

# Use shared selection from Production Explorer, fallback to NO1
pricearea = st.session_state.get("pricearea", "NO1")
lat, lon = AREA_COORDS[pricearea]

# Load hourly weather data for the chosen price area and year 2021
df = load_open_meteo_api(latitude=lat, longitude=lon, year=2021, area=pricearea)

# ---- Controls ----
options = ["All columns"] + list(df.columns)
choice = st.selectbox("Select column", options, index=0)

months = sorted(df.index.to_period("M").unique())
labels = [str(m) for m in months]

start_label, end_label = st.select_slider(
    "Select months",
    options=labels,
    value=(labels[0], labels[0]),  # default = first month
)

# ---- Subset by month range ----
start_p = pd.Period(start_label, freq="M")
end_p = pd.Period(end_label, freq="M")

mask = (df.index.to_period("M") >= start_p) & (df.index.to_period("M") <= end_p)
df_sub = df[mask]

# ---- Plot ----
if choice == "All columns":
    # Plot all variables in long format
    df_long = df_sub.reset_index().melt(
        id_vars="time", var_name="variable", value_name="value"
    )
    fig = px.line(
        df_long,
        x="time",
        y="value",
        color="variable",
        title="All variables",
        labels={"value": "Value", "time": "Time"},
    )
else:
    # Plot single selected column
    fig = px.line(
        df_sub.reset_index(),
        x="time",
        y=choice,
        title=choice,
        labels={"time": "Time", choice: choice},
    )

fig.update_layout(
    template="plotly_white",
    xaxis_title="Time",
    yaxis_title="Value",
    legend_title_text="",
    margin=dict(l=20, r=20, t=40, b=20),
    height=450,
)

st.plotly_chart(fig, use_container_width=True)
