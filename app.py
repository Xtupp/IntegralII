"""
IICG 514 - Business Intelligence | Universidad de Valparaíso
Taller: Deployment de Modelos ML con scikit-learn
------------------------------------------------------
App web con Streamlit

Para ejecutar localmente:
    pip install streamlit joblib pandas scikit-learn
    streamlit app.py

Para publicar en la web (gratis):
    1. Sube este archivo + modelo_churn.pkl + requirements.txt a GitHub
    2. Ve a https://share.streamlit.io
    3. Conecta tu repositorio y despliega siguiendo las instrucciones
"""

import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.set_page_config(page_title="Predictor de Mora", layout="centered")
st.title("Predictor de Mora de Clientes")
st.markdown("Sube un archivo CSV con datos de clientes para predecir si van a abandonar el servicio.")

st.info("""
**Columnas requeridas en el CSV:**
"edad", "antiguedad_meses", "factura_mensual_clp", "num_servicios", "reclamos_12m", "nps"
""")

# Cargar el modelo guardado
@st.cache_resource
def load_model():
    return joblib.load("modelo_mora2.pkl")

try:
    model = load_model()
    st.success("Modelo cargado correctamente")
except FileNotFoundError:
    st.error("No se encontró 'modelo_mora2.pkl'. Ejecuta primero 'modelo_base.py'.")
    st.stop()

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# Lógica principal
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())
    st.caption(f"Total de filas: {len(df)}")

    if st.button("🔮 Generar predicciones"):
        #FEATURES = ["edad", "antiguedad_meses", "ratio_factura_ingreso", "num_servicios", "indice_conflictividad", "nps"]
        FEATURES = df.drop(columns=['mora_90d'])

        # Verificar columnas
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Faltan columnas en el CSV: {missing}")
            st.stop()

        predictions   = model.predict(df.data)
        probabilities = model.predict_proba(df.data)[:, 1]  # prob. de abandono

        df["Predicción"]         = predictions
        df["Prob. Abandono (%)"] = (probabilities * 100).round(1)
        df["Resultado"]          = df["Predicción"].map({0: "✅ Se queda", 1: "🚨 Abandona"})

        st.subheader("📊 Resultados")
        st.dataframe(df[["Resultado", "Prob. Abandono (%)"] + FEATURES])

        # Resumen con métricas
        col1, col2, col3 = st.columns(3)
        total_abandona = int((predictions == 1).sum())
        total_queda    = int((predictions == 0).sum())
        pct_riesgo     = round(total_abandona / len(df) * 100, 1)

        col1.metric("🚨 En riesgo de abandono", total_abandona)
        col2.metric("✅ Se quedan", total_queda)
        col3.metric("% en riesgo", f"{pct_riesgo}%")

        # Botón para descargar resultados
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar resultados CSV", csv, "predicciones.csv", "text/csv")
