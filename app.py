import streamlit as st
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import joblib

st.title("California Home Closing Price Predictor")

model = joblib.load("xgboosting-regressor.pkl")

st.subheader("**Input Values**")
st.write("**Note:** the following features are in order of *importance* for the machine learning model XGBoosting.")

bathrooms = st.number_input("**Bathrooms**", 1, 10, step=1)

address = st.text_input("Enter Property Address (optional):", "")
latitude, longitude = None, None

if address:
    geolocator = Nominatim(user_agent="idx_home_price_app")
    location = geolocator.geocode(address)
    if location:
        latitude = location.latitude
        longitude = location.longitude
        st.success(f"Found coordinates: ({latitude}, {longitude})")
    else:
        st.error("Address not found. Please try a different address.")
if latitude is None or longitude is None:
    st.write("Click on the map to choose a location")
    m = folium.Map(location=[34.05533869827761, -118.24253732680296], zoom_start=10)
    m.add_child(folium.LatLngPopup()) # enables click capture
    
    map_data = st_folium(m, width=700)
    if map_data and map_data["last_clicked"]:
        latitude = map_data["last_clicked"]["lat"]
        longitude = map_data["last_clicked"]["lng"]
    st.write(f"**Selected Coordinates**: ({latitude}, {longitude})")

fireplace = st.selectbox("**Fireplace?**", [False, True])
    
bedrooms = st.number_input("**Bedrooms**", 0, 6, step=1)

pool = st.selectbox("**Private Pool?**", [False, True])
    
garage_spaces = st.number_input("**Garage Spaces**", 0, 6)

main_beds = st.number_input("**Main Level Bedrooms**", 0, 6)
    
total_parking = st.number_input("**Total Parking Spaces**", 0, 10)

year = st.number_input("**Year Built**", 1800, 2025)
    
view = st.selectbox("**Has View?**", [False, True])
    
stories = st.number_input("**Stories**", 1, 4) 
if stories == 2:
    levels_two = 1
    levels_multisplit = 0
elif stories >= 3:
    levels_two = 0
    levels_multisplit = 1
else:
    levels_two = 0
    levels_multisplit = 0
    
garage_attached = st.selectbox("**Attached Garage?**", [False, True])
    
new_const = st.selectbox("**New Construction?**", [False, True])


if st.button("Predict Price"):
    if latitude is None or longitude is None:
        st.error("You must provide either an address or select a location on the map.")
        st.stop()
        
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