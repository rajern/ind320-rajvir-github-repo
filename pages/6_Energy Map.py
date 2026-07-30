from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from src.data_loader import (
    load_pricearea_geojson,
    load_elhub_production_data,
    load_elhub_consumption_data,
)

# Property name in GeoJSON: properties["ElSpotOmr"] = "NO 2"
PRICEAREA_GEO_KEY = "ElSpotOmr"
VALID_PRICEAREAS = {"NO1", "NO2", "NO3", "NO4", "NO5"}


# -----------------------------
# Helpers for groups and means
# -----------------------------

def get_groups(kind: str) -> list[str]:
    """
    Return sorted list of groups for the given kind.
    kind = "production" or "consumption".
    """
    if kind == "production":
        df = load_elhub_production_data()
        field = "productiongroup"
    else:
        df = load_elhub_consumption_data()
        field = "consumptiongroup"

    groups = sorted(df[field].dropna().unique())
    return groups


def mean_by_pricearea(
    kind: str,
    group: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute mean quantity per price area for the selected kind/group/time interval.
    """
    if kind == "production":
        df = load_elhub_production_data()
        group_field = "productiongroup"
    else:
        df = load_elhub_consumption_data()
        group_field = "consumptiongroup"

    mask = (
        (df[group_field] == group)
        & (df["starttime"] >= start_ts)
        & (df["starttime"] < end_ts)
    )
    df_sel = df.loc[mask, ["pricearea", "quantitykwh"]]

    if df_sel.empty:
        return pd.DataFrame(columns=["pricearea", "mean_kwh"])

    agg = (
        df_sel.groupby("pricearea", as_index=False)["quantitykwh"]
        .mean()
        .rename(columns={"quantitykwh": "mean_kwh"})
    )
    return agg


# -----------------------------
# Map builder
# -----------------------------

def build_map(
    geojson: dict,
    df_mean: pd.DataFrame,
    selected_pricearea: str,
    clicked_coord: dict | None,
) -> folium.Map:
    """
    Build Folium map with:
    - price area polygons
    - choropleth coloring by mean_kwh
    - outline of the selected price area
    - marker for the clicked coordinate.
    """

    gj = deepcopy(geojson)

    # Map NO1..NO5 -> mean_kwh
    value_by_area: dict[str, float] = {}
    if not df_mean.empty:
        value_by_area = df_mean.set_index("pricearea")["mean_kwh"].to_dict()

    # Clean up properties: "NO 2" -> "NO2"
    for feat in gj.get("features", []):
        props = feat.setdefault("properties", {})
        pa_raw = props.get(PRICEAREA_GEO_KEY, "")
        if isinstance(pa_raw, str):
            pa_clean = pa_raw.replace(" ", "")  # "NO 2" -> "NO2"
        else:
            pa_clean = pa_raw
        props["pricearea_clean"] = pa_clean
        props["mean_kwh"] = float(value_by_area.get(pa_clean, float("nan")))

    # Reasonable center over Norway
    m = folium.Map(location=[64.5, 15.0], zoom_start=4, tiles="cartodbpositron")

    # Choropleth (uses pricearea_clean as key)
    if not df_mean.empty:
        df_choro = df_mean.rename(columns={"pricearea": "pricearea_clean"})
        folium.Choropleth(
            geo_data=gj,
            data=df_choro,
            columns=["pricearea_clean", "mean_kwh"],
            key_on="feature.properties.pricearea_clean",
            fill_color="YlOrRd",
            fill_opacity=0.6,
            nan_fill_opacity=0.0,
            line_opacity=0.0,
            legend_name="Mean quantity (kWh) in selected interval",
        ).add_to(m)

    # Outlines + tooltip, highlight selected price area
    def style_function(feature):
        pa_clean = feature["properties"].get("pricearea_clean")
        if pa_clean == selected_pricearea:
            return {"fillOpacity": 0.0, "color": "red", "weight": 3}
        else:
            return {"fillOpacity": 0.0, "color": "black", "weight": 1}

    folium.GeoJson(
        gj,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["pricearea_clean", "mean_kwh"],
            aliases=["Price area", "Mean kWh"],
            localize=True,
            sticky=False,
        ),
    ).add_to(m)

    # Marker for selected coordinate (if any)
    if clicked_coord is not None:
        folium.Marker(
            location=[clicked_coord["lat"], clicked_coord["lon"]],
            popup=(
                f"Selected point\n"
                f"lat={clicked_coord['lat']:.4f}, lon={clicked_coord['lon']:.4f}"
            ),
        ).add_to(m)

    return m


# -----------------------------
# Main page
# -----------------------------

def main():
    st.title("Energy map")

    st.caption(
        "This page shows the Norwegian price areas (NO1–NO5). "
        "You can select production or consumption, choose a group and a time "
        "interval, and see the average quantity per price area. "
        "Click on the map to store a coordinate that will later be used by "
        "the Snow Drift page."
    )

    # --- Load GeoJSON ---
    try:
        geojson = load_pricearea_geojson()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Price area from Energy Explorer (if set), otherwise default to NO2
    selected_pricearea = st.session_state.get("pricearea", "NO2")
    if selected_pricearea not in VALID_PRICEAREAS:
        selected_pricearea = "NO2"

    # --- Controls: data type, group, time interval ---
    st.subheader("Data selection")

    col1, col2 = st.columns(2)

    with col1:
        data_type_label = st.radio(
            "Data type",
            ["Production", "Consumption"],
            horizontal=True,
        )
        kind = "production" if data_type_label == "Production" else "consumption"

        groups = get_groups(kind)
        if not groups:
            st.error("No groups found in MongoDB for this data type.")
            st.stop()

        group = st.selectbox("Group", groups)

    with col2:
        min_date = date(2021, 1, 1)
        max_date = date(2024, 12, 31)

        start_date = st.date_input(
            "Start date",
            value=date(2023, 1, 1),
            min_value=min_date,
            max_value=max_date,
        )
        days = st.slider("Interval length (days)", 1, 365, 30)

    start_ts = pd.Timestamp(start_date)
    end_ts = start_ts + timedelta(days=days)

    df_mean = mean_by_pricearea(
        kind=kind,
        group=group,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    if df_mean.empty:
        st.warning("No data for this combination of type, group and time interval.")
    else:
        st.caption("Mean quantity per price area in selected interval (kWh):")
        st.dataframe(df_mean, use_container_width=True)

    # --- Map + click handling ---
    clicked_coord = st.session_state.get("map_coord")

    folium_map = build_map(
        geojson=geojson,
        df_mean=df_mean,
        selected_pricearea=selected_pricearea,
        clicked_coord=clicked_coord,
    )

    map_state = st_folium(folium_map, width=900, height=600)

    # Store last click
    if map_state and map_state.get("last_clicked"):
        click = map_state["last_clicked"]
        st.session_state["map_coord"] = {"lat": click["lat"], "lon": click["lng"]}
        st.write(
            f"Stored coordinate: lat={click['lat']:.4f}, lon={click['lng']:.4f}"
        )
    elif clicked_coord is not None:
        st.write(
            f"Current stored coordinate: "
            f"lat={clicked_coord['lat']:.4f}, lon={clicked_coord['lon']:.4f}"
        )
    else:
        st.write(
            "Click anywhere on the map to store a coordinate for the Snow Drift page."
        )


if __name__ == "__main__":
    main()
