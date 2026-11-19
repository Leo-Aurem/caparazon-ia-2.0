# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class Mensaje(BaseModel):
    mensaje: str

INTENCIONES = [
    "saludo",
    "horario",
    "ubicacion",
    "locales_libres",
    "buscar_tienda",
    "estado_incidencia",
    "cupones"
]

# Frases de ejemplo para entrenar el modelo
FRASES = [
    "hola", "buenos dias", "buenas tardes", "que tal",
    "a que hora abren", "horario", "a que hora cierran",
    "donde estan ubicados", "direccion", "donde queda",
    "hay locales libres", "espacios disponibles", "que locales puedo alquilar",
    "donde esta la tienda", "buscar tienda", "hay alguna farmacia",
    "estado de mi incidencia", "como va mi reclamo", "estado INC-2026",
    "tengo cupones", "descuentos", "mis cupones"
]

# A qué intención pertenece cada frase
ETIQUETAS = [
    "saludo", "saludo", "saludo", "saludo",
    "horario", "horario", "horario",
    "ubicacion", "ubicacion", "ubicacion",
    "locales_libres", "locales_libres", "locales_libres",
    "buscar_tienda", "buscar_tienda", "buscar_tienda",
    "estado_incidencia", "estado_incidencia", "estado_incidencia",
    "cupones", "cupones", "cupones"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(FRASES)

@app.post("/nlp")
def procesar(mensaje: Mensaje):
    user_vec = vectorizer.transform([mensaje.mensaje.lower()])
    similitudes = cosine_similarity(user_vec, X).flatten()
    mejor_idx = similitudes.argmax()
    confianza = similitudes[mejor_idx]

    if confianza < 0.35:
        return {"intencion": "desconocido", "confianza": float(confianza), "entidades": {}}

    return {
        "intencion": ETIQUETAS[mejor_idx],
        "confianza": float(confianza),
        "entidades": {} # Aquí podrías extraer códigos con expresiones regulares
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)