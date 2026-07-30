from __future__ import annotations

from copy import deepcopy

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.analysis_context import (
    AREA_LOCATIONS,
    context_caption,
    render_analysis_context,
)
from src.data_loader import (
    load_elhub_consumption_data,
    load_elhub_production_data,
    load_pricearea_geojson,
)

PRICEAREA_GEO_KEY = "ElSpotOmr"


def coordinate_from_map_state(map_state: dict | None) -> dict[str, float] | None:
    if not map_state or not map_state.get("last_clicked"):
        return None
    click = map_state["last_clicked"]
    return {"lat": float(click["lat"]), "lon": float(click["lng"])}


def coordinate_key(coord: dict[str, float]) -> tuple[float, float]:
    return round(coord["lat"], 6), round(coord["lon"], 6)


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
    marker_coord: dict | None,
    marker_label: str,
    marker_color: str,
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

    if marker_coord is not None:
        folium.Marker(
            location=[marker_coord["lat"], marker_coord["lon"]],
            popup=(
                f"{marker_label}: {marker_coord['lat']:.4f}, "
                f"{marker_coord['lon']:.4f}"
            ),
            icon=folium.Icon(color=marker_color),
        ).add_to(energy_map)

    return energy_map


def main() -> None:
    context = render_analysis_context(show_resolution=False, show_location=False)

    st.title("Energy map")
    st.caption(
        "Compare mean hourly energy across Norway's five price areas, then select a coordinate for the Snow Drift analysis."
    )
    st.markdown(
        context_caption(
            context,
            include_resolution=False,
            include_location=False,
        )
    )
    st.divider()

    try:
        geojson = load_pricearea_geojson()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    selected_pricearea = context.price_area

    st.subheader("Analysis settings")
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
    with col_2:
        group = st.selectbox("Group", groups)

    start_ts = pd.Timestamp(context.start_date)
    end_ts = pd.Timestamp(context.end_date) + pd.Timedelta(days=1)
    df_mean = mean_by_pricearea(kind, group, start_ts, end_ts)

    st.subheader("Energy comparison")
    if df_mean.empty:
        st.warning("No data are available for this type, group and time interval.")
    else:
        metric_1, metric_2 = st.columns(2)
        metric_1.metric("Areas with data", f"{len(df_mean)} of 5")
        metric_2.metric("Highest area mean", f"{df_mean['mean_gwh'].max():,.2f} GWh")
        st.caption(
            "The map compares mean hourly energy for the selected group over the shared period."
        )

    st.subheader("Choose weather location")
    st.write(
        "Click the map, review the coordinates, and confirm the point before it is "
        "used by Weather Explorer and Snow Drift. The red outline is the shared price area."
    )

    stored_coord = st.session_state.get("map_coord")
    pending_coord = st.session_state.get("pending_map_coord")
    representative_lat, representative_lon, representative_name = AREA_LOCATIONS[
        selected_pricearea
    ]

    if stored_coord is None:
        st.info(
            f"Current weather location: {representative_name} "
            f"({representative_lat:.4f}, {representative_lon:.4f}), the representative "
            f"point for {selected_pricearea}."
        )
    else:
        stored_source = st.session_state.get("location_selection_source", "map")
        source_label = "Manual coordinates" if stored_source == "manual" else "Map selection"
        st.success(
            f"Saved weather location: {stored_coord['lat']:.4f}, "
            f"{stored_coord['lon']:.4f} ({source_label})."
        )

    marker_coord = pending_coord or stored_coord
    marker_label = "Pending selection" if pending_coord else "Saved location"
    marker_color = "orange" if pending_coord else "green"
    folium_map = build_map(
        geojson=geojson,
        df_mean=df_mean,
        selected_pricearea=selected_pricearea,
        marker_coord=marker_coord,
        marker_label=marker_label,
        marker_color=marker_color,
    )
    map_state = st_folium(
        folium_map,
        height=540,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="energy_map",
    )

    clicked_coord = coordinate_from_map_state(map_state)
    if clicked_coord is not None:
        clicked_key = coordinate_key(clicked_coord)
        if clicked_key != st.session_state.get("last_processed_map_click"):
            st.session_state["last_processed_map_click"] = clicked_key
            st.session_state["pending_map_coord"] = clicked_coord
            st.rerun()

    pending_coord = st.session_state.get("pending_map_coord")
    stored_coord = st.session_state.get("map_coord")

    if pending_coord is not None:
        st.warning(
            f"Point ready to confirm: {pending_coord['lat']:.4f}, "
            f"{pending_coord['lon']:.4f}."
        )
        confirm_col, discard_col = st.columns(2)
        if confirm_col.button("Use selected point", type="primary"):
            st.session_state["map_coord"] = pending_coord
            st.session_state["location_selection_source"] = "map"
            st.session_state.pop("pending_map_coord", None)
            st.rerun()
        if discard_col.button("Discard map click"):
            st.session_state.pop("pending_map_coord", None)
            st.rerun()
    elif stored_coord is not None:
        continue_col, reset_col = st.columns(2)
        with continue_col:
            st.page_link("pages/6_Snow Drift.py", label="Continue to Snow Drift")
        if reset_col.button("Use representative location instead"):
            st.session_state.pop("map_coord", None)
            st.session_state.pop("location_selection_source", None)
            st.rerun()
    else:
        st.caption("No custom point is saved. Click the map or enter coordinates below.")

    default_coord = stored_coord or pending_coord or {
        "lat": representative_lat,
        "lon": representative_lon,
    }
    if "manual_location_latitude" not in st.session_state:
        st.session_state["manual_location_latitude"] = float(default_coord["lat"])
    if "manual_location_longitude" not in st.session_state:
        st.session_state["manual_location_longitude"] = float(default_coord["lon"])

    with st.expander("Enter coordinates manually"):
        manual_col_1, manual_col_2 = st.columns(2)
        manual_latitude = manual_col_1.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            format="%.4f",
            key="manual_location_latitude",
        )
        manual_longitude = manual_col_2.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            format="%.4f",
            key="manual_location_longitude",
        )
        if st.button("Use manual coordinates", type="primary"):
            st.session_state["map_coord"] = {
                "lat": float(manual_latitude),
                "lon": float(manual_longitude),
            }
            st.session_state["location_selection_source"] = "manual"
            st.session_state.pop("pending_map_coord", None)
            st.rerun()

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
