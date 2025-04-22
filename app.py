import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('model.pkl')

st.set_page_config(page_title="House Price Predictor", page_icon="🏡")
st.title("🏡 House Price Predictor")
st.markdown("Enter the house details to estimate its price.")

# Input features
area = st.slider('Area (in sq ft)', 500, 10000, 2000)
bedrooms = st.selectbox('Bedrooms', [1, 2, 3, 4, 5])
bathrooms = st.selectbox('Bathrooms', [1, 2, 3])
stories = st.selectbox('Stories', [1, 2, 3])
parking = st.selectbox('Parking Spots', [0, 1, 2, 3])

mainroad = st.selectbox('Main Road?', ['yes', 'no'])
guestroom = st.selectbox('Guest Room?', ['yes', 'no'])
basement = st.selectbox('Basement?', ['yes', 'no'])
hotwaterheating = st.selectbox('Hot Water Heating?', ['yes', 'no'])
airconditioning = st.selectbox('Air Conditioning?', ['yes', 'no'])
prefarea = st.selectbox('Preferred Area?', ['yes', 'no'])
furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

# Manual one-hot encoding
data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad_yes': int(mainroad == 'yes'),
    'guestroom_yes': int(guestroom == 'yes'),
    'basement_yes': int(basement == 'yes'),
    'hotwaterheating_yes': int(hotwaterheating == 'yes'),
    'airconditioning_yes': int(airconditioning == 'yes'),
    'prefarea_yes': int(prefarea == 'yes'),
    'furnishingstatus_semi-furnished': int(furnishingstatus == 'semi-furnished'),
    'furnishingstatus_unfurnished': int(furnishingstatus == 'unfurnished')
}

input_df = pd.DataFrame([data])

# Predict
if st.button('Predict Price'):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹{int(prediction):,}")
