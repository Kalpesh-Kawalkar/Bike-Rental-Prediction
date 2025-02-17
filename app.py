# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load dataset (replace 'your_dataset.csv' with your file path)
data = pd.read_csv('bike_rent.csv')

# Handle missing or non-numeric values
data.replace('?', np.nan, inplace=True)

# Split data into features and target
X = data[['registered', 'casual', 'hr', 'temp', 'atemp']]
y = data['cnt']  # Replace with your actual target column

# Use SimpleImputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Bike Rental Prediction App")

# st.image("75301.png", caption="Bike Rental Demand Prediction", use_column_width = True, width = 150)

st.write("This streamlit app predicts the total number of bikes rented per hour or day i.e. the total demand, which includes both casual users and registered users. ")

# User inputs
registered = st.number_input("Number of bikes rented by registered users (Registered Users)", min_value=0)
casual = st.number_input("Number of bikes rented by unregistered (casual) users", min_value=0)
hr = st.slider("Hour of the Day (0-23)", min_value=0, max_value=23, value=12)
temp = st.number_input("Temperature (°C)", min_value=0.0)
atemp = st.number_input("Feels Like Temperature (°C)", min_value=0.0)

# Make predictions
if st.button("Predict"):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame([[registered, casual, hr, temp, atemp]], 
                              columns=['registered', 'casual', 'hr', 'temp', 'atemp'])
    
    # Impute missing values in input data if necessary
    input_data = imputer.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"Total number of bikes rented (sum of casual and registered) : {prediction:.2f}")