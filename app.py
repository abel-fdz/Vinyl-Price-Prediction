import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json

st.set_page_config(page_title="Vinils Predictor", layout="centered")

st.title("🎵 Vinyl Price Prediction")

# ---------- Cargar modelos ----------
with open("model_rfA.pkl", "rb") as f:
    model1 = pickle.load(f)

with open("model_rfB.pkl", "rb") as f:
    model2 = pickle.load(f)

with open("model_rfC.pkl", "rb") as f:
    model3 = pickle.load(f)

# Si tienes feature_order.json
with open("feature_order.json", "r", encoding="utf-8") as f:
    FEATURE_ORDER = json.load(f)

# ---------- Inputs ----------
st.header("Vinyl characteristics")

# Opciones binarias
binarias = [
    "Limited Edition","B&W","Coloured","Splatter","Uniform","Translucid",
    "Picture Disck","Liquid","Zeotrope","Poster","Firmat","Numerat","Llibret","Defectuós","Tirada desconeguda"
]
bin_inputs = {}
for b in binarias:
    bin_inputs[b] = st.checkbox(b)

# Estado (ordinal)
mapa_estat = {"M":6,"NM":5,"VG+":4,"VG":3,"VG":3,"G":2,"P":1}
estat = st.selectbox("Estat del vinilo", options=list(mapa_estat.keys()), format_func=lambda x: f"{x} ({mapa_estat[x]})")

# Números
demandat = st.number_input("Demandat (Del 1 al 10, on 10 vol dir que el venen unes 5 persones, 8 unes 20 i etc)", min_value=0, value=1)
num_copies = st.number_input("Nº de còpies", min_value=1, value=1000)
preu_compra = st.number_input("Preu Compra Total (Contant enviament)", min_value=0.0, value=20.0)
any_vinil = st.number_input("Any de publicació", min_value=1900, max_value=2030, value=2025)

# ---------- Preparar input para el modelo ----------
data = {
    "Estat": mapa_estat[estat],
    "Demandat": demandat,
    "Nº de còpies": num_copies,
    "Preu Compra Total": preu_compra,
    "Any": any_vinil
}
data.update(bin_inputs)

# Asegurarse de tener todas las columnas
for col in FEATURE_ORDER:
    if col not in data:
        data[col] = 0

X = pd.DataFrame([data])[FEATURE_ORDER]

# ---------- Predicción ----------
if st.button("💰 Predecir precio"):
    pred1 = model1.predict(X)[0]
    pred2 = model2.predict(X)[0]
    pred3 = model3.predict(X)[0]
    
    st.success(f"**Model 1:** {pred1:.2f} €")
    st.success(f"**Model 2:** {pred2:.2f} €")
    st.success(f"**Model 3:** {pred3:.2f} €")
    
    promedio = (pred1 + pred2 + pred3) / 3
    st.info(f"**Promedio dels 3 models:** {promedio:.2f} €")
