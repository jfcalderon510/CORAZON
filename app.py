import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar el modelo y el scaler previamente entrenados
model = joblib.load('nb_model.bin')
scaler = joblib.load('scaler.pkl')

# Título de la aplicación
st.title('Herramienta para predecir si sufre del corazón')
st.subheader('Predicción de riesgo cardiovascular')
st.markdown("**Autor:** JUANFERNANDO CALDERONNN")

# Barra lateral con entradas para las variables
st.sidebar.header("Parámetros de entrada")
edad = st.sidebar.slider('Edad', min_value=20, max_value=80, value=50, step=1)
colesterol = st.sidebar.slider('Colesterol', min_value=100, max_value=600, value=200, step=10)

# Mostrar los valores que el usuario ingresa
st.sidebar.write(f"Edad: {edad}")
st.sidebar.write(f"Colesterol: {colesterol}")

# Hacer la predicción usando el modelo cargado
if st.sidebar.button("Predecir"):
    # Preparar los datos de entrada como un DataFrame
    input_data = pd.DataFrame([[edad, colesterol]], columns=['edad', 'colesterol'])

    # Escalar los datos de entrada utilizando el scaler cargado
    input_data_scaled = scaler.transform(input_data)

    # Realizar la predicción
    prediccion = model.predict(input_data_scaled)

    # Mostrar el resultado de la predicción
    if prediccion == 0:
        st.success("Resultado: **No sufrirá del corazón**")
    else:
        st.warning("Resultado: **Advertencia, sufrirá del corazón**")