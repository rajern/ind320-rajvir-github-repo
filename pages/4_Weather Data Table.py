import streamlit as st
import pandas as pd
from src.data_loader import load_open_meteo_api

st.title("Weather Data table")
st.markdown("""
    This page shows a table of weather data variables from 2021
    for the selected price area the Energy Production Explorer page.
"""
)

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

# Keep only the first month in the data
m_df = df[df.index.to_period("M") == df.index.min().to_period("M")]

# Build a small table: one row per column + sparkline data for that month
table = pd.DataFrame({
    "Variable": df.columns,
    "First month": [m_df[c].tolist() for c in df.columns],
})

# Show the table with a line chart cell for each row
st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    column_config={"First month": st.column_config.LineChartColumn("First month")},
)
