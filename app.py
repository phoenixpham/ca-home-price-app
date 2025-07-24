import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import joblib

MODEL_OPTIONS = {
    "XGBoosting Regressor": "xgboosting-regressor.pkl",
    "(Baseline) Linear Regression": "linear-regressor.pkl"
    }
default_values = {
    "bathrooms_total_integer": 2,
    "bedrooms_total": 3,
    "fireplace_yn": True,
    "levels_two": 0,
    "levels_multisplit": 0,
    "pool_private_yn": False,
    "longitude": -118.63372072401032,
    "latitude": 34.74713690524978,
    "year_built": 2024,
    "stories": 1,
    "main_level_bedrooms": 2,
    "view_yn": True,
    "new_construction_yn": False,
    "attached_garage_yn": True,
    "garage_spaces": 2,
    "parking_total": 2
    }

st.title("California Home Closing Price Predictor")
st.write("**Note:** the following features are in order of *importance* for the machine learning model.")

selected_model = st.selectbox("**Choose a prediction model**", list(MODEL_OPTIONS.keys()), index=0)
model = joblib.load(MODEL_OPTIONS[selected_model])

# inputs
bathrooms = st.number_input("**Bathrooms**", 1, 10, step=1, value=default_values["bathrooms_total_integer"])

st.write("Click on the map to choose a location")
m = folium.Map(location=[34.05533869827761, -118.24253732680296], zoom_start=10)
m.add_child(folium.LatLngPopup()) # enables click capture

map_data = st_folium(m, width=700)
if map_data and map_data["last_clicked"]:
    latitude = map_data["last_clicked"]["lat"]
    longitude = map_data["last_clicked"]["lng"]
else:
    longitude = default_values["longitude"]
    latitude = default_values["latitude"]
st.write(f"**Selected Coordinates**: ({latitude}, {longitude})")

fireplace = st.selectbox("**Fireplace?**", [False, True], index=int(default_values["fireplace_yn"]))
bedrooms = st.number_input("**Bedrooms**", 0, 6, step=1, value=default_values["bedrooms_total"])
pool = st.selectbox("**Private Pool?**", [False, True], index=int(default_values["pool_private_yn"]))
garage_spaces = st.number_input("**Garage Spaces**", 0, 6, value=default_values["garage_spaces"])
main_beds = st.number_input("**Main Level Bedrooms**", 0, 6, value=default_values["main_level_bedrooms"])
total_parking = st.number_input("**Total Parking Spaces**", 0, 10, value=default_values["parking_total"])
year = st.number_input("**Year Built**", 1800, 2025, value=default_values["year_built"])
view = st.selectbox("**Has View?**", [False, True], index=int(default_values["view_yn"]))
stories = st.number_input("**Stories**", 1, 4, value=default_values["stories"])
levels_two = st.number_input("**Levels: Two**", 0, 1, value=default_values["levels_two"])
levels_multisplit = st.number_input("**Levels: Multi-Split**", 0, 1, value=default_values["levels_multisplit"])
garage_attached = st.selectbox("**Attached Garage?**", [False, True], index=int(default_values["attached_garage_yn"]))
new_const = st.selectbox("**New Construction?**", [False, True], index=int(default_values["new_construction_yn"]))

if st.button("Predict Price"):
    input_data = {
        "bathrooms_total_integer": bathrooms,
        "bedrooms_total": bedrooms,
        "levels_two": levels_two,
        "levels_multisplit": levels_multisplit,
        "pool_private_yn": pool,
        "longitude": longitude,
        "latitude": latitude,
        "fireplace_yn": fireplace,
        "year_built": year,
        "stories": stories,
        "main_level_bedrooms": main_beds,
        "view_yn": view,
        "new_construction_yn": new_const,
        "attached_garage_yn": garage_attached,
        "garage_spaces": garage_spaces,
        "parking_total": total_parking
    }
    input_df = pd.DataFrame([input_data])

    prediction = np.expm1(model.predict(input_df)[0])  # Back-transform log price
    st.success(f"Estimated Home Closing Price: ${prediction:,.0f}")
