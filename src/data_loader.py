from pathlib import Path
import pandas as pd
import streamlit as st
from pymongo import MongoClient
import requests

# Cache function for loading the Open-Meteo subset data
@st.cache_data(show_spinner=False)
def load_open_meteo() -> pd.DataFrame:
    data_path = Path(__file__).parent.parent / "data" / "open-meteo-subset.csv"
    df = pd.read_csv(data_path, parse_dates=["time"])
    df.set_index("time", inplace=True)
    return df

# Cache function for loading production data from MongoDB (elhub2021 / production_per_group_hour)
@st.cache_data
def load_elhub_production_data() -> pd.DataFrame:
    """
    Load production_per_group_hour from MongoDB (all years).
    """
    client = MongoClient(st.secrets["MONGODB_URI"])
    db = client["elhub2021"]
    col = db["production_per_group_hour"]

    records = list(col.find({}, {"_id": 0}))
    client.close()

    df = pd.DataFrame(records)
    df["starttime"] = pd.to_datetime(df["starttime"])
    return df

# Cache function for loading consumption data from MongoDB (elhub2021 / consumption_per_group_hour)
@st.cache_data
def load_elhub_consumption_data() -> pd.DataFrame:
    """
    Load consumption_per_group_hour from MongoDB (all years).
    """
    client = MongoClient(st.secrets["MONGODB_URI"])
    db = client["elhub2021"]
    col = db["consumption_per_group_hour"]

    records = list(col.find({}, {"_id": 0}))
    client.close()

    df = pd.DataFrame(records)
    df["starttime"] = pd.to_datetime(df["starttime"])
    return df

# Cache function for loading Open-Meteo data from the API
@st.cache_data(show_spinner=False)
def load_open_meteo_api(
    latitude: float,
    longitude: float,
    year: int = 2021, # Choose default year as 2021
    area: str | None = None,
) -> pd.DataFrame:
    """Download hourly weather data for given coordinates and year."""
    url = "https://archive-api.open-meteo.com/v1/archive"

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ],
        "models": "era5",
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

# Cache function for loading price area GeoJSON file
@st.cache_data(show_spinner=False)
def load_pricearea_geojson() -> dict:
    """
    Load the price area GeoJSON file from data/file.geojson.
    """
    geo_path = Path(__file__).parent.parent / "data" / "file.geojson"
    with geo_path.open("r", encoding="utf-8") as f:
        return json.load(f)