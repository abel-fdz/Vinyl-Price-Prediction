import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

#streamlit run app.py
import pickle
from sklearn.ensemble import RandomForestRegressor


# ------------------ Llegir dades ------------------
datos = pd.read_csv("vinils_taula_estadistica.csv", delimiter=";", decimal=",")

# ------------------ Columnes binàries ------------------
binarias = [
    "Limited Edition","B&W","Coloured","Splatter","Uniform",
    "Translucid","Picture Disck", "Liquid", "Zeotrope","Poster","Firmat","Numerat",
    "Llibret","Defectuós", "Tirada desconeguda"
]

# Convertir CERT/FALS → 1/0
for col in binarias:
    datos[col] = datos[col].map({"CERT": 1, "FALS": 0})

# ------------------ Columna tirada desconeguda i Nº de còpies ------------------
datos["Nº de còpies"] = pd.to_numeric(datos["Nº de còpies"], errors="coerce")
datos["Demandat"] = pd.to_numeric(datos["Demandat"], errors="coerce")
datos["Tirada desconeguda"] = datos["Nº de còpies"].isna().astype(int)  # 1 si desconeguda, 0 si coneguda
datos["Nº de còpies"] = datos["Nº de còpies"].fillna(1_000_000)
  # valor gran per indicar incertesa

# ------------------ Altres columnes numèriques ------------------
for col in ["Preu Compra Total", "Any", "Preu mercat"]:
    datos[col] = pd.to_numeric(datos[col], errors="coerce")

# ------------------ Afegir columna ESTAT com a ordinal ------------------
mapa_estat = {
    "M": 6,     # Mint
    "NM": 5,    # Near Mint
    "VG+": 4,   # Very Good Plus
    "VG": 3,    # Very Good
    "F": 3,
    "G": 2,     # Good
    "P": 1      # Poor
}

datos["Estat"] = datos["Estat"].map(mapa_estat)

# ------------------ Detectar valors no numèrics ------------------
for col in ["Nº de còpies", "Preu Compra Total", "Preu mercat", "Any"]:
    no_convertibles = datos[pd.to_numeric(datos[col], errors="coerce").isna()][col]
    if not no_convertibles.empty:
        print(f"\nColumna '{col}' té valors no numèrics:")
        print(no_convertibles)

# ------------------ Variables predictives i target ------------------
X = datos[binarias + ["Estat", "Demandat", "Nº de còpies", "Preu Compra Total", "Any"]]
y = datos["Preu mercat"]

# ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# ------------------ MODEL 1 ------------------
modelo1 = RandomForestRegressor(n_estimators=10, max_features=1, random_state=123, n_jobs=-1)
modelo1.fit(X_train, y_train)
preds1 = modelo1.predict(X_test)
rmse1 = root_mean_squared_error(y_test, preds1)
print(f"MODELO 1 RMSE: {rmse1:.4f}")

# ------------------ MODEL 2 ------------------
modelo2 = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=123, n_jobs=-1)
modelo2.fit(X_train, y_train)
preds2 = modelo2.predict(X_test)
rmse2 = root_mean_squared_error(y_test, preds2)
print(f"MODELO 2 RMSE: {rmse2:.4f}")

# ------------------ Exemple: predir nou vinil ------------------
nou_vinil = pd.DataFrame([{
    # "Obert": 0,
    # "Obert amb plàstic original": 0,
    # "Venia Sense Plastic": 0,
    # "No tocat mai": 1,
    "Estat": mapa_estat["NM"],   # Exemple: Near Mint,
    "Demandat": 9,
    "Limited Edition": 1,
    "B&W": 0,
    "Coloured": 1,
    "Splatter": 0,
    "Uniform": 0,
    "Translucid": 0,
    "Picture Disck": 0,
    "Liquid": 1,
    "Zeotrope": 0,
    "Poster": 0,
    "Firmat": 0,
    "Numerat": 0,
    "Llibret": 0,
    "Defectuós": 0,
    "Nº de còpies": 4000,
    "Tirada desconeguda": 0,
    "Preu Compra Total": 51,
    "Any": 2025
}])

nou_vinil = nou_vinil[X_train.columns]
prediccio1 = modelo1.predict(nou_vinil)[0]
prediccio2 = modelo2.predict(nou_vinil)[0]

print(f"\nPredicció Model 1: {prediccio1:.2f} €")
print(f"Predicció Model 2: {prediccio2:.2f} €")


# Guardar model amb pickle
with open("model_rf1.pkl", "wb") as f:
    pickle.dump(modelo1, f)

print("Model guardat correctament a 'model_rf1.pkl'")

# Guardar model amb pickle
with open("model_rf2.pkl", "wb") as f:
    pickle.dump(modelo2, f)

print("Model guardat correctament a 'model_rf2.pkl'")


# train_and_export.py (exemple)
import pickle, json
# ... després d’entrenar el teu model com a 'modelo' i crear X_train ...

with open("model_rf1.pkl", "wb") as f:
    pickle.dump(modelo1, f)

with open("feature_order.json", "w", encoding="utf-8") as f:
    json.dump(list(X_train.columns), f, ensure_ascii=False, indent=2)

print("✅ Guardats: model_rf.pkl i feature_order.json")
