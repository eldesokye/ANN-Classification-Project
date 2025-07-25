import streamlit as st 
import tensorflow as tf 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import numpy as np 


## load the trained model 

model = load_model('salary_regression_model.h5')


# load the encoders and scaler

with open('labelencoder2.pkl', 'rb') as file:
    label_encoder= pickle.load(file)

with open('ohe2.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

with open('scaler2.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app

st.title('Salary Prediction App')

st.markdown('This app predicts the salary of an employee based on their information.')
st.markdown('Please enter the following details:')

# user input 

# Input fields
geography = st.selectbox('Geography', onehot_encoder.categories_[0].tolist())
gender = st.selectbox('Gender', label_encoder.classes_.tolist())
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1])
tenure = st.number_input('Tenure (in years)', min_value=0, max_value=10, value=1)
number_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})


# one hot encode the geography
# One-hot encode the geography
geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Concatenate the one-hot encoded geography with the input data
input_data = pd.concat([input_data.reset_index(drop=True ), geo_encoded_df], axis=1)


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)

prediction_salary = prediction[0][0]


st.write(f"The predicted salary is: ${prediction_salary:,.2f}")