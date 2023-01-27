# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:37:29 2023

@author: Fayis
"""

import streamlit as st
import pandas as pd
import pickle

# Load your trained model
with open('linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(data:pd.DataFrame):
    # Use the model to predict the mark
    result = model.predict(data)

    # return the result
    return result

st.title("Student Mark Predictor")

file_upload = st.file_uploader("Upload a csv file", type=["csv"])

if file_upload is not None:
    data = pd.read_csv(file_upload)
    if st.button("Predict"):
        result = predict(data)
        st.success('The predicted marks are {}'.format(result))
