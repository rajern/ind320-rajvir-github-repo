import streamlit as st
import pandas as pd
import plotly.express as px

from src.analysis_context import context_caption, render_analysis_context
from src.data_loader import load_elhub_production_data

context = render_analysis_context(show_location=False)

st.title("Energy explorer")
st.caption(
    "Explore electricity production by price area, period and production group."
)
st.markdown(context_caption(context, include_location=False))

# Load Elhub production data from MongoDB via cached helper
df = load_elhub_production_data()
area = context.price_area
start_ts = pd.Timestamp(context.start_date)
end_ts = pd.Timestamp(context.end_date) + pd.Timedelta(days=1)
resolution_rule = {"Hourly": "h", "Daily": "D", "Monthly": "MS"}[
    context.resolution
]

df_period = df[
    (df["pricearea"] == area)
    & (df["starttime"] >= start_ts)
    & (df["starttime"] < end_ts)
]

if df_period.empty:
    st.warning("No production data found for the active selection.")
    st.stop()

groups_in_area = sorted(
    df_period["productiongroup"].dropna().unique().tolist()
)

st.divider()
st.subheader("Analysis settings")
selected_groups = st.pills(
    "Groups shown in the time series",
    options=groups_in_area,
    selection_mode="multi",
    default=groups_in_area,
)
st.caption("The production-share chart includes all available groups in the selected period.")

st.subheader("Results")

# Split the layout into two columns
left_col, right_col = st.columns(2)

# ---- Left: production share for the selected period ----
with left_col:
    st.subheader("Share by group")

    pie_data = (
        df_period
        .groupby("productiongroup", as_index=False)["quantitykwh"]
        .sum()
        .assign(productiongroup=lambda data: data["productiongroup"].str.title())
        .assign(gwh=lambda data: data["quantitykwh"] / 1_000_000)
        .sort_values("gwh", ascending=True)
    )

    total_gwh = pie_data["gwh"].sum()
    dominant = pie_data.iloc[-1]
    metric_1, metric_2 = st.columns(2)
    metric_1.metric("Total production", f"{total_gwh:,.0f} GWh")
    metric_2.metric(
        "Largest group",
        dominant["productiongroup"],
        f"{dominant['gwh'] / total_gwh:.0%} of total",
    )

    fig_pie = px.bar(
        pie_data,
        x="gwh",
        y="productiongroup",
        orientation="h",
        labels={"gwh": "Production (GWh)", "productiongroup": ""},
        text_auto=".3s",
    )
    fig_pie.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    fig_pie.update_layout(
        title=(
            f"{area} | {context.start_date:%d %b} - "
            f"{context.end_date:%d %b %Y}"
        )
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ---- Right: time series by production group ----
with right_col:
    st.subheader("Production over time")

    # Filter by the shared context and chosen groups
    mask = (
        (df["pricearea"] == area)
        & (df["starttime"] >= start_ts)
        & (df["starttime"] < end_ts)
    )
    if selected_groups:
        mask &= df["productiongroup"].isin(selected_groups)
    else:
        mask &= False

    # Aggregate energy using the shared time resolution
    df_series = (
        df[mask]
        .groupby(
            [
                pd.Grouper(key="starttime", freq=resolution_rule),
                "productiongroup",
            ],
            as_index=False,
        )["quantitykwh"]
        .sum()
        .assign(
            productiongroup=lambda data: data["productiongroup"].str.title(),
            gwh=lambda data: data["quantitykwh"] / 1_000_000,
        )
    )

    if df_series.empty:
        st.info("No production data for this selection.")
    else:
        fig_line = px.line(
            df_series,
            x="starttime",
            y="gwh",
            color="productiongroup",
            labels={"starttime": "Time", "gwh": "Production (GWh)", "productiongroup": "Group"},
        )
        fig_line.update_layout(
            title=f"{area} | {context.resolution} production",
            xaxis_title="Time",
            yaxis_title="Production (GWh)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=70, b=20),
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption(
            f"Values are aggregated to {context.resolution.lower()} totals for each production group."
        )

# ---- Expander with short documentation ----
with st.expander("About"):
    st.write(
        "Data: Elhub 2021 (production per group and price area). "
        "Period, price area and time resolution are controlled from the shared sidebar."
    )
