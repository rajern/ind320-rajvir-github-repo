from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import streamlit as st


PRICE_AREAS = ("NO1", "NO2", "NO3", "NO4", "NO5")
RESOLUTIONS = ("Hourly", "Daily", "Monthly")
MIN_DATE = date(2021, 1, 1)
MAX_DATE = date(2021, 12, 31)

AREA_LOCATIONS = {
    "NO1": (59.91390, 10.75220, "Oslo"),
    "NO2": (58.14670, 7.99560, "Kristiansand"),
    "NO3": (63.43050, 10.39510, "Trondheim"),
    "NO4": (69.64920, 18.95600, "Tromso"),
    "NO5": (60.39299, 5.32415, "Bergen"),
}


@dataclass(frozen=True)
class AnalysisContext:
    price_area: str
    start_date: date
    end_date: date
    resolution: str
    latitude: float
    longitude: float
    location_label: str
    location_source: str


def _initialise_state() -> None:
    if st.session_state.get("pricearea") not in PRICE_AREAS:
        st.session_state["pricearea"] = PRICE_AREAS[0]

    if st.session_state.get("analysis_resolution") not in RESOLUTIONS:
        st.session_state["analysis_resolution"] = "Daily"

    period = st.session_state.get("analysis_period")
    if not (
        isinstance(period, (tuple, list))
        and len(period) == 2
        and all(isinstance(value, date) for value in period)
    ):
        st.session_state["analysis_period"] = (MIN_DATE, MAX_DATE)


def _weather_location(price_area: str) -> tuple[float, float, str, str]:
    map_coord = st.session_state.get("map_coord")
    if (
        isinstance(map_coord, dict)
        and isinstance(map_coord.get("lat"), (int, float))
        and isinstance(map_coord.get("lon"), (int, float))
    ):
        latitude = float(map_coord["lat"])
        longitude = float(map_coord["lon"])
        source = st.session_state.get("location_selection_source", "map")
        if source == "manual":
            return latitude, longitude, "Manual coordinates", "manual"
        return latitude, longitude, "Selected map point", "map"

    latitude, longitude, label = AREA_LOCATIONS[price_area]
    return latitude, longitude, label, "representative"


def render_analysis_context(
    *,
    show_period: bool = True,
    show_resolution: bool = True,
    show_location: bool = True,
) -> AnalysisContext:
    """Render the shared sidebar controls and return their current values."""
    _initialise_state()

    st.sidebar.header("Analysis context")
    price_area = st.sidebar.selectbox(
        "Price area",
        PRICE_AREAS,
        key="pricearea",
        help="Shared by the energy and weather views.",
    )
    if show_period:
        period = st.sidebar.date_input(
            "Period",
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            key="analysis_period",
            help="The pilot currently uses the common 2021 data period.",
        )
    else:
        period = st.session_state["analysis_period"]
    if show_resolution:
        resolution = st.sidebar.selectbox(
            "Time resolution",
            RESOLUTIONS,
            key="analysis_resolution",
        )
    else:
        resolution = st.session_state["analysis_resolution"]

    if isinstance(period, (tuple, list)) and len(period) == 2:
        start_date, end_date = period
    else:
        start_date = end_date = period[0] if period else MIN_DATE
        if show_period:
            st.sidebar.caption("Choose an end date to complete the period.")

    latitude, longitude, location_label, location_source = _weather_location(price_area)

    if show_location:
        st.sidebar.markdown("**Weather location**")
        st.sidebar.caption(f"{location_label} | {latitude:.4f}, {longitude:.4f}")
        if location_source == "representative":
            st.sidebar.caption("Representative point for the selected price area.")
        elif location_source == "manual":
            st.sidebar.caption("Coordinates entered manually on the Energy Map.")
        else:
            st.sidebar.caption("Point selected on the Energy Map.")

    return AnalysisContext(
        price_area=price_area,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        latitude=latitude,
        longitude=longitude,
        location_label=location_label,
        location_source=location_source,
    )


def context_caption(
    context: AnalysisContext,
    *,
    include_resolution: bool = True,
    include_location: bool = True,
) -> str:
    period = (
        f"{context.start_date:%d %b %Y}"
        if context.start_date == context.end_date
        else f"{context.start_date:%d %b %Y} - {context.end_date:%d %b %Y}"
    )
    parts = [context.price_area, period]
    if include_resolution:
        parts.append(context.resolution.lower())
    if include_location:
        parts.append(context.location_label)
    return f"**Active selection:** {' | '.join(parts)}"
