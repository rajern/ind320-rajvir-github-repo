# Norwegian Energy & Weather Analytics

An interactive Streamlit application for exploring relationships between Norwegian electricity production, consumption and weather conditions.

**[Open the live application](https://ind320-rajvir-app-repo-cmhz46bwwk9apw8zvbaxa2.streamlit.app/)**

## Overview

This project combines Norwegian energy data with historical weather observations in a multi-page analytical dashboard. It supports exploratory analysis, anomaly detection, geospatial comparison, snow-drift estimation and short-term energy forecasting.

The application is designed around Norway's five electricity price areas and uses shared Streamlit state so that selections can carry across several views.

## Key features

- Explore hourly electricity production by price area and production group.
- Decompose energy time series with STL and inspect frequency patterns with spectrograms.
- Retrieve and visualize historical ERA5 weather data from Open-Meteo.
- Detect temperature and precipitation anomalies using statistical process control and Local Outlier Factor.
- Compare energy and weather variables through lagged and rolling correlations.
- Explore energy data geographically with interactive price-area maps.
- Estimate snow transport and visualize directional exposure.
- Generate configurable SARIMAX-based energy forecasts.

## Technical design

- Multi-page Streamlit interface with grouped navigation and a shared analysis context.
- Cached data loaders for MongoDB and the Open-Meteo Historical Weather API.
- Interactive Plotly charts and Folium maps with consistent cross-page selections.
- Statistical methods including STL, spectrograms, SPC, Local Outlier Factor, rolling correlations and SARIMAX.
- Explicit empty, loading and error states for analyses that depend on external services.

## Data sources

- **Elhub:** Norwegian electricity production and consumption data, accessed from a MongoDB collection prepared for the project.
- **Open-Meteo Historical Weather API:** hourly ERA5 weather variables including temperature, precipitation, wind speed, wind direction and wind gusts.
- **NVE / GeoNorge:** geographic boundaries for Norwegian electricity price areas.

External data remains the property of its respective providers and may be subject to separate terms of use.

## Project structure

```text
Home.py                            Streamlit entry point and navigation
pages/                             Interactive analysis pages
src/analysis_context.py            Shared filters and cross-page state
src/data_loader.py                 Cached data access and API integrations
src/Snow_drift.py                  Snow-transport calculations
data/                              Weather subset and price-area geometry
.streamlit/config.toml             Shared visual theme
.streamlit/secrets.example.toml    Safe configuration example
requirements.txt                   Python dependencies
```

## Run locally

### Prerequisites

- Python 3.10 or newer
- Access to a MongoDB instance containing the required Elhub collections

### Installation

```bash
git clone https://github.com/rajern/norwegian-energy-weather-analytics.git
cd norwegian-energy-weather-analytics
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`, then replace the placeholder with your MongoDB connection string:

```toml
MONGODB_URI = "mongodb+srv://<username>:<password>@<cluster>/"
```

Do not commit this file or expose the connection string publicly.

Start the application:

```bash
streamlit run Home.py
```

## Analytical scope and limitations

- Most analyses use data from 2021 and should not be interpreted as a current view of the Norwegian energy system.
- Results depend on the availability and quality of external APIs and the configured MongoDB database.
- The forecasting view is intended for exploration and demonstration, not operational planning or trading.
- Weather coordinates represent selected locations within each price area rather than every location in the area.

## Project background

The project originated as coursework in the NMBU course IND320 and was subsequently reorganized as an independent portfolio project. The implementation and analysis are presented for demonstration and learning purposes.
