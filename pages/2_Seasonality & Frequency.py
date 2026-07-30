import streamlit as st
from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis_context import context_caption, render_analysis_context
from src.data_loader import load_elhub_production_data

context = render_analysis_context(show_resolution=False, show_location=False)

st.title("Seasonality & frequency")
st.caption(
    "Separate recurring seasonal patterns from trends and inspect the frequency content of energy production."
)
st.markdown(
    context_caption(
        context,
        include_resolution=False,
        include_location=False,
    )
)
st.caption("These analyses use hourly observations within the shared period.")

# ---- Analysis helpers ----

# Function for STL decomposition (LOESS) on Elhub production data
def plot_stl_elhub(
    df,
    area="NO3",
    group="hydro",
    period=24,
    seasonal=13,
    trend=365,
    robust=True,
    time_col="starttime",
    value_col="quantitykwh",
):
    # Filter for the chosen price area and production group
    sub = df[(df["pricearea"] == area) & (df["productiongroup"] == group)].copy()
    if sub.empty:
        raise ValueError("No data for this price area and production group.")

    # Make sure we have a sorted datetime index
    sub[time_col] = pd.to_datetime(sub[time_col])
    sub = sub.sort_values(time_col)
    series = sub.set_index(time_col)[value_col]

    # Remove duplicate indices if any
    series = series[~series.index.duplicated(keep="first")]

    # Force regular hourly frequency and fill small gaps
    series = series.asfreq("h")
    series = series.interpolate(limit_direction="both")

    # Run STL (Seasonal-Trend decomposition using LOESS)
    stl = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        robust=robust,
    )
    result = stl.fit()

    # ---- Plotly decomposition figure ----
    idx = series.index

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
    )

    fig.add_trace(
        go.Scatter(x=idx, y=result.observed, name="Observed", line=dict(width=1)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=idx, y=result.trend, name="Trend", line=dict(width=1)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=idx, y=result.seasonal, name="Seasonal", line=dict(width=1)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=idx, y=result.resid, name="Residual", line=dict(width=1)),
        row=4,
        col=1,
    )

    fig.update_layout(
        title=f"STL decomposition – {area}, {group}",
        height=800,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig, result


# Function for creating a spectrogram from Elhub production data
def plot_spectrogram_elhub(
    df,
    area="NO4",
    group="hydro",
    window_length=24 * 7,   # window length in hours
    window_overlap=0.5,     # fraction of overlap between windows
):

    # Filter data for selected price area and production group
    mask = (df["pricearea"] == area) & (df["productiongroup"] == group)
    df_sel = df.loc[mask].sort_values("starttime")

    if df_sel.empty:
        raise ValueError("No data for this price area and production group.")

    # Extract production series as numpy array
    x = df_sel["quantitykwh"].to_numpy(dtype=float)

    # Convert window parameters to integers for spectrogram
    nperseg = int(window_length)
    noverlap = int(window_length * window_overlap)

    # Handle short series gracefully
    if len(x) < nperseg:
        nperseg = max(1, len(x))
        noverlap = int(nperseg * window_overlap)

    # Compute spectrogram using SciPy
    f, t, Sxx = spectrogram(x, fs=1.0, nperseg=nperseg, noverlap=noverlap)

    # Avoid log of zero
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            x=t,
            y=f,
            z=Sxx_db,
            colorbar=dict(title="Power (dB)"),
        )
    )
    fig.update_layout(
        title=f"Spectrogram – {area}, {group}",
        xaxis_title="Time window",
        yaxis_title="Frequency [cycles per hour]",
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Second return value kept for compatibility with existing unpacking
    return fig, (f, t, Sxx)


# ---- Load Elhub data for the shared price area and period ----

df_all = load_elhub_production_data()
current_area = context.price_area
start_ts = pd.Timestamp(context.start_date)
end_ts = pd.Timestamp(context.end_date) + pd.Timedelta(days=1)
df = df_all[
    (df_all["starttime"] >= start_ts)
    & (df_all["starttime"] < end_ts)
].copy()

if df.loc[df["pricearea"] == current_area].empty:
    st.warning("No production data are available for the shared area and period.")
    st.stop()

# All production groups in this price area
groups = sorted(
    df.loc[df["pricearea"] == current_area, "productiongroup"]
      .dropna()
      .unique()
      .tolist()
)

st.divider()
st.subheader("Choose analysis")

# ---- Tabs for STL and Spectrogram ----
tab_stl, tab_spec = st.tabs(["STL decomposition", "Spectrogram"])

with tab_stl:
    st.write("Separate the observed series into trend, recurring seasonality and residual variation.")

    st.subheader("Analysis settings")
    group = st.selectbox("Production group", groups)
    with st.expander("Advanced settings"):
        period = st.number_input("Period (hours)", min_value=1, value=24, step=1)
        seasonal = st.number_input("Seasonal smoother", min_value=3, value=13, step=1)
        trend = st.number_input("Trend smoother", min_value=3, value=365, step=1)
        robust = st.checkbox("Use robust fitting", value=True)

    run_stl = st.button("Run STL analysis", type="primary")
    if not run_stl:
        st.info("Choose a production group and run the analysis. Default settings represent a daily pattern.")
    else:
        try:
            with st.spinner("Computing STL decomposition..."):
                fig_stl, _ = plot_stl_elhub(
                    df,
                    area=current_area,
                    group=group,
                    period=period,
                    seasonal=seasonal,
                    trend=trend,
                    robust=robust,
                )
            st.subheader("Results")
            st.plotly_chart(fig_stl, use_container_width=True)
            st.caption("Observed = trend + seasonal component + residual. Large residuals are variation not captured by the selected pattern.")
        except ValueError as exc:
            st.warning(str(exc))

with tab_spec:
    st.write("Inspect how the strength of recurring frequencies changes over time.")

    st.subheader("Analysis settings")
    group_spec = st.selectbox("Production group", groups, key="spec_group")
    with st.expander("Advanced settings"):
        window_length = st.number_input(
            "Window length (hours)",
            min_value=24,
            max_value=24 * 60,
            value=24 * 7,
            step=24,
        )
        window_overlap = st.slider(
            "Window overlap",
            min_value=0.0,
            max_value=0.9,
            value=0.5,
            step=0.1,
        )

    run_spec = st.button("Run frequency analysis", type="primary")
    if not run_spec:
        st.info("Run the analysis to display the frequency pattern for the selected production group.")
    else:
        try:
            with st.spinner("Computing spectrogram..."):
                fig_spec, _ = plot_spectrogram_elhub(
                    df,
                    area=current_area,
                    group=group_spec,
                    window_length=window_length,
                    window_overlap=window_overlap,
                )
            st.subheader("Results")
            st.plotly_chart(fig_spec, use_container_width=True)
            st.caption("Brighter bands indicate frequencies with stronger energy during that part of the year.")
        except ValueError as exc:
            st.warning(str(exc))
