import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_elhub_production_data

st.title("Energy explorer")
st.caption(
    "Explore hourly electricity production by price area, month and production group."
)

# Load Elhub production data from MongoDB via cached helper
df = load_elhub_production_data()
YEAR = 2021

# Available price areas
AREAS = sorted(df["pricearea"].dropna().unique().tolist())

# Make sure we have a shared price area in session state
if "pricearea" not in st.session_state:
    st.session_state["pricearea"] = AREAS[0]

# Radio buttons for price area (central selector for the whole app)
area = st.radio(
    "Price area",
    AREAS,
    index=AREAS.index(st.session_state["pricearea"]),
    horizontal=True,
)

# Update session state whenever user changes selection
st.session_state["pricearea"] = area

# Split the layout into two columns
left_col, right_col = st.columns(2)

# ---- Left: pie chart for 2021 ----
with left_col:
    st.subheader("Share by group (2021)")

    # Filter to selected area and year
    df_area_2021 = df[
        (df["pricearea"] == area)
        & (df["starttime"].dt.year == YEAR)
    ]

    if df_area_2021.empty:
        st.warning("No data found for 2021.")
    else:
        # Aggregate energy per production group
        pie_data = (
            df_area_2021
            .groupby("productiongroup", as_index=False)["quantitykwh"]
            .sum()
            .rename(columns={"quantitykwh": "kwh"})
            .sort_values("kwh", ascending=False)
        )

        # Simple pie chart of share by group
        fig_pie = px.pie(
            pie_data,
            names="productiongroup",
            values="kwh",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(title=f"{area} – {YEAR}")
        st.plotly_chart(fig_pie, use_container_width=True)

# ---- Right: line plot per month and production group ----
with right_col:
    st.subheader("Hourly lines by month")

    # Simple month selector (1–12)
    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        format_func=lambda m: f"{m:02d}",
    )

    # All production groups that exist in this price area
    groups_in_area = sorted(
        df.loc[df["pricearea"] == area, "productiongroup"]
        .dropna()
        .unique()
        .tolist()
    )

    # Pills for selecting one or more production groups
    selected_groups = st.pills(
        "Production group(s)",
        options=groups_in_area,
        selection_mode="multi",
        default=groups_in_area,
    )

    # Filter by area, year, month and chosen groups
    mask = (
        (df["pricearea"] == area)
        & (df["starttime"].dt.year == YEAR)
        & (df["starttime"].dt.month == month)
    )
    if selected_groups:
        mask &= df["productiongroup"].isin(selected_groups)

    # Aggregate hourly kWh per group (one line per group)
    df_hourly = (
        df[mask]
        .groupby(["starttime", "productiongroup"], as_index=False)["quantitykwh"]
        .sum()
        .rename(columns={"quantitykwh": "kwh"})
    )

    if df_hourly.empty:
        st.info("No hourly data for this selection.")
    else:
        fig_line = px.line(
            df_hourly,
            x="starttime",
            y="kwh",
            color="productiongroup",
        )
        fig_line.update_layout(
            title=f"{area} – {YEAR}-{month:02d}",
            xaxis_title="Time",
            yaxis_title="kWh",
        )
        st.plotly_chart(fig_line, use_container_width=True)

# ---- Expander with short documentation ----
with st.expander("About"):
    st.write("Data: Elhub 2021 (production per group and price area).")
