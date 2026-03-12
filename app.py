import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

#loading scaler
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('label_encoder_sex.pkl')

#giving title to the app
st.title('Titanic Survival Prediction')
st.write('Enter the details of the passenger to predict whether they survived or not.')

# Create input fields for user to enter passenger details
pclass = st.selectbox('Passenger Class (1, 2, 3)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=1, max_value=80, value=30)
sibsp = st.slider('Number of Siblings/Spouses Aboard',0, 10, 0)
parch = st.slider('Number of Parents/Children Aboard',0, 10, 0)
fare = st.slider('Fare', min_value=0.0, max_value=512.0, value=32.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Encode the categorical variables
sex  = encoder.transform([sex])[0]
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0


# Prepare the input data for prediction
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S]])
input_data_scaled = scaler.transform(input_data)

#predicting the survival
if st.button('Predict Survival'):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success('The passenger is predicted to have survived.')
    else:
        st.error('The passenger is predicted to have died.')

         
