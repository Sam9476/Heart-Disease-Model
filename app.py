from joblib import load
import numpy as np
import streamlit as st

# Load the saved model and scaler
model = load('heart_disease_model.pkl')  # Load the trained model
scaler = load('scalar.pkl')  # Load the scaler used for training

# Streamlit App Layout
st.title("Heart Disease Prediction App")
st.write("Enter the details to predict if a person has heart disease or not:")

# Input fields for user data (age, sex, etc.)
age = st.slider("Age", 29, 77, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 250)
fbs = st.selectbox("Fasting Blood Sugar (> 120 mg/dl) (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", 71, 202, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.slider("Old Peak Depression", 0.0, 6.2, 1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Collect user input into a feature vector
user_input = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

# Display user input
st.write("User Input Features:")
st.write(f"Age: {age}, Sex: {sex}, Chest Pain Type: {cp}, Resting Blood Pressure: {trestbps}, "
         f"Cholesterol: {chol}, Fasting Blood Sugar: {fbs}, Resting Electrocardiogram: {restecg}, "
         f"Max Heart Rate: {thalach}, Exercise Induced Angina: {exang}, Old Peak Depression: {oldpeak}, "
         f"Slope: {slope}, Major Vessels: {ca}, Thalassemia: {thal}")

# Make prediction when the button is clicked
if st.button("Predict Heart Disease"):
    # Scale the input features using the same scaler used for training
    user_input_scaled = scaler.transform([user_input])  # Scale the input features
    
    # Make prediction using the model
    prediction = model.predict(user_input_scaled)  # Get model prediction
    
    # Show prediction result
    if prediction > 0.5:
        st.success(f"Prediction: **Heart Disease**")
    else:
        st.success(f"Prediction: **No Heart Disease**")