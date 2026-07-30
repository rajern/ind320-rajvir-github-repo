import streamlit as st

from src.analysis_context import context_caption, render_analysis_context


context = render_analysis_context()

st.title("Norwegian Energy & Weather Analytics")
st.markdown(
    "Explore how weather conditions, geography and seasonality relate to "
    "electricity production and consumption across Norway's five price areas."
)

st.info(
    "**Data scope:** Most views focus on historical data from 2021. "
    "The dashboard is an analytical portfolio project, not a live operational system."
)
st.markdown(context_caption(context))

st.subheader("What you can explore")

energy_col, weather_col = st.columns(2)

with energy_col:
    st.markdown("### Energy system")
    st.write(
        "Explore production profiles, compare price areas on a map and build "
        "configurable energy forecasts."
    )

with weather_col:
    st.markdown("### Weather and risk")
    st.write(
        "Inspect historical weather, identify unusual observations and estimate "
        "snow transport for a selected location."
    )

with st.container():
    st.markdown("### Patterns and relationships")
    st.write(
        "Study seasonality, frequency patterns and lagged relationships between "
        "weather and the energy system."
    )

st.subheader("Getting started")
st.markdown(
    """
1. Set the shared price area, period and time resolution under **Analysis context**.
2. Open **Energy Explorer** or **Weather Explorer** to test the shared selection.
3. A point selected on **Energy Map** replaces the representative weather location.
    """
)

st.divider()
st.caption(
    "Data sources: Elhub energy data, Open-Meteo ERA5 weather data and "
    "NVE / GeoNorge price-area boundaries."
)
st.markdown(
    "[GitHub repository](https://github.com/rajern/norwegian-energy-weather-analytics) "
    "| [Live application](https://ind320-rajvir-app-repo-cmhz46bwwk9apw8zvbaxa2.streamlit.app/)"
)
