# app.py
import streamlit as st
import pandas as pd
import pickle

# ---------- Carregar model ----------
with open("model_rf1.pkl", "rb") as f:
    modelo = pickle.load(f)

# ---------- Mapes per convertir valors ----------
map_bin = {"CERT": 1, "FALS": 0}
map_estat = {"M": 6, "NM": 5, "VG+": 4, "VG": 3, "G": 2, "P": 1}

# ---------- Títol ----------
st.title("Predicció del preu d'un vinil")

st.write("Introdueix les característiques del vinil per predir el seu preu de mercat estimat.")

# ---------- Entrades ----------
cols_bin = [
    "Obert","Obert amb plàstic original","Venia Sense Plastic","No tocat mai",
    "Limited Edition","B&W","Coloured","Splatter","Uniform",
    "Translucid","Picture Disck","Liquid","Zeotrope","Poster","Firmat",
    "Numerat","Llibret","Defectuós","Tirada desconeguda"
]

input_data = {}

# Inputs binaris
for col in cols_bin:
    input_data[col] = st.selectbox(col, ["CERT", "FALS"], index=1 if col=="Tirada desconeguda" else 0)

# Inputs numèrics i ordinals
input_data["Estat"] = st.selectbox("Estat", ["M","NM","VG+","VG","G","P"])
input_data["Demandat"] = st.number_input("Demandat", min_value=0, value=1)
input_data["Nº de còpies"] = st.number_input("Nº de còpies", min_value=0, value=1000)
input_data["Preu Compra Total"] = st.number_input("Preu Compra Total (€)", min_value=0.0, value=20.0)
input_data["Any"] = st.number_input("Any", min_value=1900, max_value=2100, value=2025)

# ---------- Convertir a DataFrame i map binary/ordinal ----------
for col in cols_bin:
    input_data[col] = map_bin[input_data[col]]
input_data["Estat"] = map_estat[input_data["Estat"]]

nou_vinil = pd.DataFrame([input_data])

# ---------- Predicció ----------
if st.button("Predir preu"):
    pred = modelo.predict(nou_vinil)[0]
    st.success(f"Preu estimat: {pred:.2f} €")
