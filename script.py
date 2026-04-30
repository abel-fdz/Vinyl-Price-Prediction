import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error
import json
import pickle
import numpy as np

# ══════════════════════════════════════════════════════════════
#  1. LLEGIR DADES
# ══════════════════════════════════════════════════════════════
print("[1/9] Llegint dades...")
datos = pd.read_csv("vinils_taula_estadistica.csv", delimiter=";", decimal=",")


def parse_eu_number(series):
    """Converteix valors numèrics europeus sense trencar decimals ja correctes."""
    def _parse_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if text in {"", "nan", "NaN", "None"}:
            return None
        if "," in text:
            text = text.replace(".", "").replace(",", ".")
        elif text.count(".") >= 1:
            chunks = text.split(".")
            if all(part.isdigit() for part in chunks) and all(len(part) == 3 for part in chunks[1:]):
                text = "".join(chunks)
        try:
            return float(text)
        except ValueError:
            return None
    return series.apply(_parse_value)


# ══════════════════════════════════════════════════════════════
#  2. COLUMNES BINÀRIES
# ══════════════════════════════════════════════════════════════
print("[2/9] Convertint columnes binàries...")
binarias = [
    "Limited Edition", "B&W", "Coloured", "Splatter", "Uniform",
    "Translucid", "Picture Disck", "Liquid", "Zeotrope", "Poster",
    "Firmat", "Numerat", "Llibret", "Defectuós", "Tirada desconeguda"
]

for col in binarias:
    datos[col] = datos[col].map({"CERT": 1, "FALS": 0})


# ══════════════════════════════════════════════════════════════
#  3. COLUMNES NUMÈRIQUES
# ══════════════════════════════════════════════════════════════
print("[3/9] Netejant columnes numèriques (format europeu)...")
datos["Nº de còpies"] = parse_eu_number(datos["Nº de còpies"])
datos["Demandat"] = parse_eu_number(datos["Demandat"])
datos["Tirada desconeguda"] = datos["Nº de còpies"].isna().astype(int)
datos["Nº de còpies"] = datos["Nº de còpies"].fillna(1_000_000)

for col in ["Preu Compra Total", "Any", "Preu mercat"]:
    datos[col] = parse_eu_number(datos[col])


# ══════════════════════════════════════════════════════════════
#  4. COLUMNA ESTAT (ordinal)
# ══════════════════════════════════════════════════════════════
print("[4/9] Codificant columna 'Estat'...")
mapa_estat = {
    "M":   6,
    "NM":  5,
    "VG+": 4,
    "VG":  3,
    "F":   3,
    "G":   2,
    "P":   1
}
datos["Estat"] = datos["Estat"].map(mapa_estat)


# ══════════════════════════════════════════════════════════════
#  5. DETECTAR VALORS NO NUMÈRICS
# ══════════════════════════════════════════════════════════════
for col in ["Nº de còpies", "Preu Compra Total", "Preu mercat", "Any"]:
    no_convertibles = datos[datos[col].isna()][col]
    if not no_convertibles.empty:
        print(f"\nColumna '{col}' té valors no numèrics:")
        print(no_convertibles)


# ══════════════════════════════════════════════════════════════
#  6. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
print("[5/9] Preparant train/test split...")
X = datos[binarias + ["Estat", "Demandat", "Nº de còpies", "Preu Compra Total", "Any"]]
y = datos["Preu mercat"]

valid_mask = y.notna()
if (~valid_mask).any():
    print(f"S'eliminen {int((~valid_mask).sum())} files amb 'Preu mercat' buit/no numèric.")
X = X.loc[valid_mask]
y = y.loc[valid_mask]
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=123)

print(f"  → Dataset total:      {len(X)} vinils")
print(f"  → Entrenament:        {len(X_train)} vinils")
print(f"  → Test:               {len(X_test)} vinils")
print(f"  → Features:           {X.shape[1]}")


# ══════════════════════════════════════════════════════════════
#  7. DEFINICIÓ DE MODELS
# ══════════════════════════════════════════════════════════════

# ── Model 1 (original, per comparar) ──────────────────────────
modelo1 = RandomForestRegressor(
    n_estimators=10,
    max_depth=8,
    max_features=1,
    random_state=123,
    n_jobs=-1
)

# ── Model 2 (original, per comparar) ──────────────────────────
modelo2 = RandomForestRegressor(
    n_estimators=50,
    max_depth=8,
    max_features="sqrt",
    random_state=123,
    n_jobs=-1
)

# ── Model 3 (original, per comparar) ──────────────────────────
modelo3 = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    max_features="sqrt",
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=123,
    n_jobs=-1
)

# ── Model A — Baseline robusta ─────────────────────────────────
# max_features="sqrt" + molts arbres per estabilitat
modelo_A = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    max_features="sqrt",
    random_state=123,
    n_jobs=-1
)

# ── Model B — Conservador (antioverfitting) ────────────────────
# Profunditat baixa + min_samples_leaf per no memoritzar vinils rars
modelo_B = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    max_features="sqrt",
    min_samples_leaf=3,   # cada fulla necessita mínim 3 exemples
    random_state=123,
    n_jobs=-1
)

# ── Model C — Configuració robusta per dataset petit ──────────
# max_samples=0.8 dona diversitat extra entre arbres
modelo_C = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    max_features="sqrt",
    min_samples_split=8,  # no dividir si el node té menys de 8 mostres
    min_samples_leaf=4,   # fulles amb mínim 4 mostres
    max_samples=0.8,      # cada arbre veu el 80% de les dades
    random_state=123,
    n_jobs=-1
)


# ══════════════════════════════════════════════════════════════
#  8. ENTRENAMENT I AVALUACIÓ
# ══════════════════════════════════════════════════════════════
print("\n[6/9] Entrenant i avaluant tots els models...\n")

def avaluar_model(nom, model, X_train, X_test, y_train, y_test):
    """Entrena el model i imprimeix RMSE train/test i ràtio d'overfitting."""
    model.fit(X_train, y_train)

    preds_train = model.predict(X_train)
    preds_test  = model.predict(X_test)

    rmse_train = root_mean_squared_error(y_train, preds_train)
    rmse_test  = root_mean_squared_error(y_test,  preds_test)
    ratio      = rmse_train / rmse_test  # ideal: ~1.0

    print(f"  {'─'*42}")
    print(f"  {nom}")
    print(f"    RMSE train : {rmse_train:8.2f} €")
    print(f"    RMSE test  : {rmse_test:8.2f} €")
    print(f"    Ràtio      : {ratio:.2f}  {'✓ bo' if ratio > 0.7 else '⚠ possible overfitting'}")

    return preds_test, rmse_test

print("  Models originals (referència):")
preds1, rmse1 = avaluar_model("Model 1 (original)", modelo1, X_train, X_test, y_train, y_test)
preds2, rmse2 = avaluar_model("Model 2 (original)", modelo2, X_train, X_test, y_train, y_test)
preds3, rmse3 = avaluar_model("Model 3 (original)", modelo3, X_train, X_test, y_train, y_test)

print("\n  Models nous:")
preds_A, rmse_A = avaluar_model("Model A (baseline robusta)", modelo_A, X_train, X_test, y_train, y_test)
preds_B, rmse_B = avaluar_model("Model B (conservador)",      modelo_B, X_train, X_test, y_train, y_test)
preds_C, rmse_C = avaluar_model("Model C (robust petit)",     modelo_C, X_train, X_test, y_train, y_test)

print(f"\n  {'═'*42}")
millor_rmse = min(rmse1, rmse2, rmse3, rmse_A, rmse_B, rmse_C)
noms = {rmse1: "Model 1", rmse2: "Model 2", rmse3: "Model 3",
        rmse_A: "Model A", rmse_B: "Model B", rmse_C: "Model C"}
print(f"  Millor model (RMSE test): {noms[millor_rmse]} → {millor_rmse:.2f} €")


# ══════════════════════════════════════════════════════════════
#  8b. CROSS-VALIDATION (més fiable que un sol test_size)
# ══════════════════════════════════════════════════════════════
print("\n[7/9] Cross-validation 5-fold (més fiable)...\n")

for nom, model in [("Model A", modelo_A), ("Model B", modelo_B), ("Model C", modelo_C)]:
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    rmses = -scores
    print(f"  {nom}:  RMSE mitjà = {rmses.mean():.2f} €  (±{rmses.std():.2f})")


# ══════════════════════════════════════════════════════════════
#  8c. IMPORTÀNCIA DE FEATURES (model C com a referència)
# ══════════════════════════════════════════════════════════════
print("\n[8/9] Importància de features (Model C):\n")
importances = pd.Series(modelo_C.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)
for feat, imp in importances_sorted.items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<22} {imp:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════
#  9. EXEMPLE DE PREDICCIÓ
# ══════════════════════════════════════════════════════════════
print("\n[9/9] Predicció d'exemple (vinil nou)...\n")

nou_vinil = pd.DataFrame([{
    "Estat":              mapa_estat["NM"],
    "Demandat":           9,
    "Limited Edition":    1,
    "B&W":                0,
    "Coloured":           1,
    "Splatter":           0,
    "Uniform":            0,
    "Translucid":         0,
    "Picture Disck":      0,
    "Liquid":             1,
    "Zeotrope":           0,
    "Poster":             0,
    "Firmat":             0,
    "Numerat":            0,
    "Llibret":            0,
    "Defectuós":          0,
    "Nº de còpies":       4000,
    "Tirada desconeguda": 0,
    "Preu Compra Total":  51,
    "Any":                2025
}])

nou_vinil = nou_vinil[X_train.columns]

for nom, model in [
    ("Model 1 (original)", modelo1),
    ("Model 2 (original)", modelo2),
    ("Model 3 (original)", modelo3),
    ("Model A",            modelo_A),
    ("Model B",            modelo_B),
    ("Model C",            modelo_C),
]:
    pred = model.predict(nou_vinil)[0]
    print(f"  {nom:<26} → {pred:.2f} €")


# ══════════════════════════════════════════════════════════════
#  GUARDAR MODELS I FEATURE ORDER
# ══════════════════════════════════════════════════════════════
models_a_guardar = {
    "model_rf1.pkl": modelo1,
    "model_rf2.pkl": modelo2,
    "model_rf3.pkl": modelo3,
    "model_rfA.pkl": modelo_A,
    "model_rfB.pkl": modelo_B,
    "model_rfC.pkl": modelo_C,
}

for filename, model in models_a_guardar.items():
    with open(filename, "wb") as f:
        pickle.dump(model, f)

with open("feature_order.json", "w", encoding="utf-8") as f:
    json.dump(list(X_train.columns), f, ensure_ascii=False, indent=2)

print("\nModels guardats: model_rf1..3.pkl, model_rfA..C.pkl, feature_order.json")
print("Fet!")