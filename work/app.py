import streamlit as st
import numpy as np
import pickle

# LOAD MODELS

lr_model = pickle.load(open("Linear_Regression_Model.pkl", "rb"))
dt_model = pickle.load(open("Decision_Tree_Model.pkl", "rb"))
rf_model = pickle.load(open("Random_Forest_Model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Bike Demand Predictor", layout="centered")

st.title("Bike Sharing Demand Prediction")
st.write("Select a model to predict bike demand.")

#----------------------------

# MODEL SELECTOR

model_choice = st.selectbox(
    "Choose Machine Learning Model",
    ["Linear Regression", "Decision Tree", "Random Forest (Best Model)"]
)

if model_choice == "Linear Regression":
    model = lr_model
elif model_choice == "Decision Tree":
    model = dt_model
else:
    model = rf_model

#----------------------------

# USER INPUTS

# Season

season_label = st.selectbox("Season", ["Spring", "Summer", "Fall (Autumn)", "Winter"])
season_map = {"Spring": 1,  "Summer": 2, "Fall (Autumn)": 3, "Winter": 4}
season = season_map[season_label]


# Year

year_label = st.selectbox("Year", ["2011", "2012"])
yr = 0 if year_label == "2011" else 1


# Month

month = st.slider("Month", min_value=1, max_value=12)


# Hour

hr = st.slider("Hour of Day", min_value=0, max_value=23)


# Holiday

holiday_label = st.selectbox("Is it a Public Holiday?", ["No", "Yes"])
holiday = 1 if holiday_label == "Yes" else 0


# Weekday

weekday_label = st.selectbox("Day of the Week", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])

weekday_map = {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}
weekday = weekday_map[weekday_label]


# Working Day

workingday_label = st.selectbox("Working Day", ["Weekend / Holiday", "Normal Working Day"])
workingday = 1 if workingday_label == "Normal Working Day" else 0


# Weather Situation 

weather_label = st.selectbox("Weather Condition", ["Clear / Few Clouds", "Mist / Cloudy", "Light Rain / Snow", "Heavy Rain / Snow"])

weather_map = {"Clear / Few Clouds": 1, "Mist / Cloudy": 2, "Light Rain / Snow": 3, "Heavy Rain / Snow": 4}
weathersit = weather_map[weather_label]

# Environmental Factors 

temp = st.slider("Temperature",min_value=0.0,max_value=1.0)
atemp = st.slider("Feels-like Temperature",min_value=0.0,max_value=1.0)
hum = st.slider("Humidity",min_value=0.0,max_value=1.0)
windspeed = st.slider("Windspeed",min_value=0.0,max_value=1.0)

#---------------------------

# PREDICTION

input_data = np.array([[
    season, yr, month, hr, holiday, weekday,
    workingday, weathersit, temp, atemp, hum, windspeed
]])

input_data = input_data[:, [features.index(f) for f in features]]

if st.button("Predict Bike Demand"):
    prediction = model.predict(input_data)
    st.success(
        f"Predicted Bike Demand ({model_choice}): {int(prediction[0])} Bikes"
    )
