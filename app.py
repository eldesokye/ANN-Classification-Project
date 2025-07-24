import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pickle
import streamlit as st
# load the trained model 
model  = tf.keras.models.load_model('model.h5')


## load the encoder and scaler
with open ('labelencoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)


with open ('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)


with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## streamlit app 

st.title('Customer churn prediction')
st.markdown('This app predicts whether a customer will churn based on their information.')
st.markdown('Please enter the following details:')
# Input fields
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0].tolist())
gender = st.selectbox('Gender', label_encoder_gender.classes_.tolist())
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure (in years)', min_value=0, max_value=10, value=1)
number_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate the one-hot encoded geography with the input data
input_data = pd.concat([input_data.reset_index(drop=True ), geo_encoded_df], axis=1)


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)


# Prediction prob 
prediction_prob = prediction[0][0]


if prediction_prob > 0.5:
    st.write(f"Prediction Probability: {prediction_prob:.2f}")
    st.write("the customer is likely to churn ")

else : 
    st.write(f"Prediction Probability: {prediction_prob:.2f}")
    st.write("the customer is not likely to churn ")