# 🎵 Vinyl Price Predictor

Aquest projecte utilitza Machine Learning per estimar el valor de mercat d'un disc de vinil a partir de les seves característiques tècniques i el seu estat de conservació. 

D'un hobby com és col·leccionar vinils, n'ha sortit una eina útil per a col·leccionistes que volen saber quin és el preu real de la seva preuada col·lecció.

## 🚀 L'Aplicació
Pots provar el predictor en viu aquí: [vinyl-price-prediction.streamlit.app](https://vinyl-price-prediction.streamlit.app)

## 🧐 Com funciona?
El model ha estat entrenat amb un dataset propi recollit de col·leccions personals i preus de mercat actualitzats. L'usuari només ha d'introduir dades com:
- **Estat del vinil (Media Condition)**
- **Estat de la caràtula (Sleeve Condition)**
- **Any d'edició**
- **Raresa o gènere**

L'algoritme processa aquestes variables i retorna un **preu estimat de venda** basat en les tendències actuals.

## 🛠️ Stack Tecnològic
- **Llenguatge:** Python 🐍
- **Web App:** Streamlit
- **Biblioteques de ML:** Pandas, NumPy, RandomForest
