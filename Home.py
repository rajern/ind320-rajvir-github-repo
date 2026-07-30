import streamlit as st

st.set_page_config(
    page_title="Norwegian Energy & Weather Analytics",
    page_icon="\u26A1",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("## Energy & Weather")
st.sidebar.caption("Norway analytics portfolio")
st.sidebar.divider()

pages = {
    "Overview": [
        st.Page("pages/0_Overview.py", title="Project overview", default=True),
    ],
    "Energy": [
        st.Page("pages/1_Energy Explorer.py", title="Energy explorer"),
        st.Page(
            "pages/2_Seasonality & Frequency.py",
            title="Seasonality & frequency",
        ),
        st.Page("pages/5_Energy Map.py", title="Energy map"),
    ],
    "Weather": [
        st.Page("pages/3_Weather Explorer.py", title="Weather explorer"),
        st.Page("pages/4_Anomaly Detection.py", title="Anomaly detection"),
        st.Page("pages/6_Snow Drift.py", title="Snow drift"),
    ],
    "Weather & energy": [
        st.Page(
            "pages/7_Weather-Energy Relationships.py",
            title="Relationships",
        ),
    ],
    "Forecasting": [
        st.Page("pages/8_Energy Forecasting.py", title="Energy forecasting"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
