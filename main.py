from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import pickle, json, os

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "model_rf.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "feature_order.json")

# ---------- Carregar model i ordre de columnes ----------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURE_ORDER = json.load(f)

# ---------- Esquema d'entrada (totes les features requerides) ----------
class VinylInput(BaseModel):
    Estat: int = Field(..., ge=1, le=6)                 # 1..6 (P..M)
    Demandat: float = Field(..., ge=0)
    Limited_Edition: int = Field(..., ge=0, le=1)
    B_W: int = Field(..., ge=0, le=1)
    Coloured: int = Field(..., ge=0, le=1)
    Splatter: int = Field(..., ge=0, le=1)
    Uniform: int = Field(..., ge=0, le=1)
    Translucid: int = Field(..., ge=0, le=1)
    Picture_Disck: int = Field(..., ge=0, le=1)
    Liquid: int = Field(..., ge=0, le=1)
    Zeotrope: int = Field(..., ge=0, le=1)
    Poster: int = Field(..., ge=0, le=1)
    Firmat: int = Field(..., ge=0, le=1)
    Numerat: int = Field(..., ge=0, le=1)
    Llibret: int = Field(..., ge=0, le=1)
    Defectuós: int = Field(..., ge=0, le=1)
    _N_de_còpies: float = Field(..., alias="Nº de còpies", ge=0)
    Tirada_desconeguda: int = Field(..., ge=0, le=1)
    Preu_Compra_Total: float = Field(..., ge=0)
    Any: float = Field(..., ge=1800, le=3000)

    class Config:
        populate_by_name = True  # permet rebre "Nº de còpies" al JSON

app = FastAPI(title="Vinils Predictor API", version="1.0.0")

# CORS obert (canvia origins a la teva app web si vols restringir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: VinylInput):
    try:
        # Convertir a dict amb aliases (perquè "Nº de còpies" ho respecti)
        data = payload.model_dump(by_alias=True)
        # Construir DataFrame i reordenar exactament com al training
        df = pd.DataFrame([data])
        # Ajust de noms per coincidir amb FEATURE_ORDER exactament
        # (per exemple, si al training tens "Limited Edition" amb espai)
        rename_map = {
            "Limited_Edition": "Limited Edition",
            "B_W": "B&W",
            "Picture_Disck": "Picture Disck",
            "_N_de_còpies": "Nº de còpies",
            "Tirada_desconeguda": "Tirada desconeguda",
            "Preu_Compra_Total": "Preu Compra Total",
        }
        df = df.rename(columns=rename_map)

        # Verificar que totes les columnes existeixen
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Falten columnes: {missing}")

        df = df[FEATURE_ORDER]

        y = model.predict(df)[0]
        return {"prediction": float(y)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
