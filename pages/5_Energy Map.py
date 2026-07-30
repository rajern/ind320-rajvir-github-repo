from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.data_loader import (
    load_elhub_consumption_data,
    load_elhub_production_data,
    load_pricearea_geojson,
)

PRICEAREA_GEO_KEY = "ElSpotOmr"
VALID_PRICEAREAS = {"NO1", "NO2", "NO3", "NO4", "NO5"}


def get_groups(kind: str) -> list[str]:
    if kind == "production":
        df = load_elhub_production_data()
        field = "productiongroup"
    else:
        df = load_elhub_consumption_data()
        field = "consumptiongroup"
    return sorted(df[field].dropna().unique())


def mean_by_pricearea(
    kind: str,
    group: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
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
    selected = df.loc[mask, ["pricearea", "quantitykwh"]]
    if selected.empty:
        return pd.DataFrame(columns=["pricearea", "mean_gwh"])

    return (
        selected.groupby("pricearea", as_index=False)["quantitykwh"]
        .mean()
        .assign(mean_gwh=lambda data: data["quantitykwh"] / 1_000_000)
        [["pricearea", "mean_gwh"]]
    )


def build_map(
    geojson: dict,
    df_mean: pd.DataFrame,
    selected_pricearea: str,
    clicked_coord: dict | None,
) -> folium.Map:
    geojson_copy = deepcopy(geojson)
    values = (
        df_mean.set_index("pricearea")["mean_gwh"].to_dict()
        if not df_mean.empty
        else {}
    )

    for feature in geojson_copy.get("features", []):
        properties = feature.setdefault("properties", {})
        raw_area = properties.get(PRICEAREA_GEO_KEY, "")
        clean_area = raw_area.replace(" ", "") if isinstance(raw_area, str) else raw_area
        properties["pricearea_clean"] = clean_area
        properties["mean_gwh"] = float(values.get(clean_area, float("nan")))

    energy_map = folium.Map(
        location=[64.5, 15.0],
        zoom_start=4,
        tiles="cartodbpositron",
    )

    if not df_mean.empty:
        choropleth_data = df_mean.rename(columns={"pricearea": "pricearea_clean"})
        folium.Choropleth(
            geo_data=geojson_copy,
            data=choropleth_data,
            columns=["pricearea_clean", "mean_gwh"],
            key_on="feature.properties.pricearea_clean",
            fill_color="YlOrRd",
            fill_opacity=0.65,
            nan_fill_opacity=0.0,
            line_opacity=0.0,
            legend_name="Mean hourly energy (GWh)",
        ).add_to(energy_map)

    def style_function(feature):
        area = feature["properties"].get("pricearea_clean")
        if area == selected_pricearea:
            return {"fillOpacity": 0.0, "color": "#d62728", "weight": 3}
        return {"fillOpacity": 0.0, "color": "#333333", "weight": 1}

    folium.GeoJson(
        geojson_copy,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["pricearea_clean", "mean_gwh"],
            aliases=["Price area", "Mean hourly energy (GWh)"],
            localize=True,
            sticky=False,
        ),
    ).add_to(energy_map)

    if clicked_coord is not None:
        folium.Marker(
            location=[clicked_coord["lat"], clicked_coord["lon"]],
            popup=(
                f"Selected point: {clicked_coord['lat']:.4f}, "
                f"{clicked_coord['lon']:.4f}"
            ),
        ).add_to(energy_map)

    return energy_map


def main() -> None:
    st.title("Energy map")
    st.caption(
        "Compare mean hourly energy across Norway's five price areas, then select a coordinate for the Snow Drift analysis."
    )

    try:
        geojson = load_pricearea_geojson()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    selected_pricearea = st.session_state.get("pricearea", "NO2")
    if selected_pricearea not in VALID_PRICEAREAS:
        selected_pricearea = "NO2"

    st.subheader("Data selection")
    col_1, col_2 = st.columns(2)
    with col_1:
        data_type_label = st.radio(
            "Data type",
            ["Production", "Consumption"],
            horizontal=True,
        )
        kind = "production" if data_type_label == "Production" else "consumption"
        groups = get_groups(kind)
        if not groups:
            st.error("No groups were found for this data type.")
            st.stop()
        group = st.selectbox("Group", groups)

    with col_2:
        start_date = st.date_input(
            "Start date",
            value=date(2023, 1, 1),
            min_value=date(2021, 1, 1),
            max_value=date(2024, 12, 31),
        )
        days = st.slider("Interval length (days)", 1, 365, 30)

    start_ts = pd.Timestamp(start_date)
    end_ts = start_ts + timedelta(days=days)
    df_mean = mean_by_pricearea(kind, group, start_ts, end_ts)

    if df_mean.empty:
        st.warning("No data are available for this type, group and time interval.")
    else:
        metric_1, metric_2 = st.columns(2)
        metric_1.metric("Areas with data", f"{len(df_mean)} of 5")
        metric_2.metric("Highest area mean", f"{df_mean['mean_gwh'].max():,.2f} GWh")

    st.subheader("Map and location selection")
    st.write("Click the map to store a coordinate. The red outline is the price area selected in Energy Explorer.")

    clicked_coord = st.session_state.get("map_coord")
    folium_map = build_map(
        geojson=geojson,
        df_mean=df_mean,
        selected_pricearea=selected_pricearea,
        clicked_coord=clicked_coord,
    )
    map_state = st_folium(
        folium_map,
        height=540,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="energy_map",
    )

    if map_state and map_state.get("last_clicked"):
        click = map_state["last_clicked"]
        st.session_state["map_coord"] = {"lat": click["lat"], "lon": click["lng"]}
        clicked_coord = st.session_state["map_coord"]

    if clicked_coord is None:
        st.info("No coordinate is stored yet. Click the map before opening Snow Drift.")
    else:
        st.success(
            f"Stored coordinate: {clicked_coord['lat']:.4f}, {clicked_coord['lon']:.4f}"
        )
        st.page_link("pages/6_Snow Drift.py", label="Continue to Snow Drift")

    if not df_mean.empty:
        with st.expander("View underlying area values"):
            display_table = df_mean.rename(
                columns={"pricearea": "Price area", "mean_gwh": "Mean hourly energy (GWh)"}
            )
            st.dataframe(
                display_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Mean hourly energy (GWh)": st.column_config.NumberColumn(format="%.2f")
                },
            )


if __name__ == "__main__":
    main()
