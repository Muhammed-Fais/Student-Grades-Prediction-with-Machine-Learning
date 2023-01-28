# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:37:29 2023

@author: Fayis
"""

import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load your trained model
with open('linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(data:pd.DataFrame):
    # Use the model to predict the mark
    result = model.predict(data)

    # return the result
    return result

st.title("Student Mark Predictor")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit Bank Note Authenticator ML App </h2>
</div>
"""

Grade1 = st.number_input("Enter Grade1 out of 100")
Grade2 = st.number_input("Enter Grade2 out of 100")
Study_Time = st.number_input("Enter Study Time in hours per week")
Failures = st.number_input("Enter the number of past class failures")
Absence = st.number_input("Enter the number of school absence days")

data = {'Grade1':[Grade1],'Grade2':[Grade2],'Study_Time':[Study_Time],'Failures':[Failures],'Absence':[Absence]}
data = pd.DataFrame(data)

if st.button("Predict"):
    result = predict(data)
    st.success('The predicted mark is {}'.format(result))






