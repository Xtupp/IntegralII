"""
IICG 514 - Business Intelligence | Universidad de Valparaíso
Predicción de Riesgo de Morosidad en ConecTel S.A.

"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import io

app = FastAPI(
    title="API Predictor de Clientes en Mora",
    description="Recibe datos de clientes y retorna predicciones de abandono.",
    version="1.0.0"
)

# Cargar el modelo al iniciar la API
try:
    model = joblib.load("modelo_mora2.pkl")
except FileNotFoundError:
    raise RuntimeError("No se encontró 'modelo_mora2.pkl'. Ejecuta primero modelo_base.py")

FEATURES = ["edad", "tipo_contrato", "antiguedad_meses", "plan", "tiene_internet", "velocidad_mbps", "tiene_tv", "tiene_linea_movil", "num_servicios", "ratio_factura_ingreso", "dias_mora_hist", "indice_conflictividad", "nps", "descuento_activo", "meses_sin_reajuste", "region_Araucanía", "region_Atacama", "region_Biobío", "region_Coquimbo", "region_Los Lagos", "region_Maule", "region_Metropolitana", "region_O'Higgins", "region_Valparaíso", "metodo_pago_Débito automático", "metodo_pago_Efectivo", "metodo_pago_Transferencia", "metodo_pago_WebPay"]

# Definición del esquema de entrada
class ClienteInput(BaseModel):
    edad: int
    tipo_contrato: str
    antiguedad_meses: int
    plan: str
    tiene_internet: int
    velocidad_mbps: float
    tiene_tv: int
    tiene_linea_movil: int
    num_servicios: int
    ratio_factura_ingreso: float
    dias_mora_hist: int
    indice_conflictividad: float
    nps: int
    descuento_activo: int
    meses_sin_reajuste: int

    region_Araucanía: int
    region_Atacama: int
    region_Biobío: int
    region_Coquimbo: int
    region_Los_Lagos: int = Field(alias="region_Los Lagos")
    region_Maule: int
    region_Metropolitana: int
    region_O_Higgins: int = Field(alias="region_O'Higgins")
    region_Valparaíso: int

    metodo_pago_Débito_automático: int = Field(alias="metodo_pago_Débito automático")
    metodo_pago_Efectivo: int
    metodo_pago_Transferencia: int
    metodo_pago_WebPay: int

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "edad": 35,
                    "tipo_contrato": "postpago",
                    "antiguedad_meses": 24,
                    "plan": "fibra_600",
                    "tiene_internet": 1,
                    "velocidad_mbps": 600.0,
                    "tiene_tv": 1,
                    "tiene_linea_movil": 0,
                    "num_servicios": 2,
                    "ratio_factura_ingreso": 0.08,
                    "dias_mora_hist": 3,
                    "indice_conflictividad": 0.2,
                    "nps": 8,
                    "descuento_activo": 1,
                    "meses_sin_reajuste": 6,
                    "region_Araucanía": 0,
                    "region_Atacama": 0,
                    "region_Biobío": 0,
                    "region_Coquimbo": 0,
                    "region_Los Lagos": 0,
                    "region_Maule": 0,
                    "region_Metropolitana": 1,
                    "region_O'Higgins": 0,
                    "region_Valparaíso": 0,
                    "metodo_pago_Débito automático": 1,
                    "metodo_pago_Efectivo": 0,
                    "metodo_pago_Transferencia": 0,
                    "metodo_pago_WebPay": 0
                }
            ]
        }
    }

# Endpoint 1: Verificación de estado
@app.get("/")
def read_root():
    return {"status": "ok", "mensaje": "API de prediccion de Mora funcionando"}

# Endpoint 2: Predecir UN cliente con JSON
@app.post("/predecir")
def predecir_cliente(cliente: ClienteInput):
    """
    Recibe los datos de UN cliente en formato JSON
    y retorna si abandonará o no el servicio.
    """
    df = pd.DataFrame([cliente.model_dump()])

    prediccion   = int(model.predict(df[FEATURES])[0])
    probabilidad = round(float(model.predict_proba(df[FEATURES])[0][1]), 2)

    return {
        "prediccion":    prediccion,
        "resultado":     "Abandona" if prediccion == 1 else "Se queda",
        "prob_abandono": probabilidad
    }

# Endpoint 3: Predecir MÚLTIPLES clientes con CSV
@app.post("/predecir-csv")
async def predecir_csv(file: UploadFile = File(...)):
    """
    Recibe un archivo CSV con múltiples clientes
    y retorna las predicciones para todos.
    """
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columnas faltantes en el CSV: {missing}")

    predicciones   = model.predict(df[FEATURES])
    probabilidades = model.predict_proba(df[FEATURES])[:, 1]

    df["prediccion"]    = predicciones
    df["resultado"]     = pd.Series(predicciones).map({0: "Se queda", 1: "Abandona"}).values
    df["prob_abandono"] = probabilidades.round(2)

    return df.to_dict(orient="records")

    #Nota:
    ##orient especifica la forma del diccionario que devuelve DataFrame.to_dict().
    #Con orient="records" obtienes una lista de diccionarios, uno por fila (cada dict mapea columna→valor).
    
