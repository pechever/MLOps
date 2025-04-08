#app.py

import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo entrenado
def load_model():
    with open('modelo_entrenado.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title('Predicción con Modelo ML')

feature_input = st.number_input('Ingrese el valor de la característica:')

if st.button('Predecir'):
    if feature_input is not None:
        data = pd.DataFrame({'feature': [feature_input]})
        prediction = model.predict(data)[0]
        st.success(f'La predicción es: {prediction:.2f}')
    else:
        st.warning('Por favor, ingrese un valor para la característica.')