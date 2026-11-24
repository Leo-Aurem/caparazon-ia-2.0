# -*- coding: utf-8 -*-
import re
import random
import unicodedata
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Caparazón IA - Asistente Conversacional")

class Mensaje(BaseModel):
    mensaje: str

def normalizar_texto(texto: str) -> str:
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

# Dataset ampliado con lenguaje natural y conversacional
DATASET = [
    # Saludos
    ("hola", "saludo"),
    ("buenos dias", "saludo"),
    ("buenas tardes", "saludo"),
    ("buenas noches", "saludo"),
    ("que tal como estas", "saludo"),
    ("hola necesito ayuda", "saludo"),
    ("hey hola", "saludo"),

    # Horarios
    ("a que hora abren", "horario"),
    ("cual es el horario de atencion", "horario"),
    ("a que hora cierran el mall", "horario"),
    ("atienden los domingos y feriados", "horario"),
    ("hasta que hora esta abierto", "horario"),
    ("horarios del centro comercial", "horario"),

    # Ubicación
    ("donde estan ubicados", "ubicacion"),
    ("cual es la direccion exacta", "ubicacion"),
    ("donde queda el mall caparazon", "ubicacion"),
    ("como llego al centro comercial", "ubicacion"),
    ("en que zona se encuentran", "ubicacion"),

    # Locales libres / Alquiler
    ("hay locales libres", "locales_libres"),
    ("tienen espacios disponibles para alquilar", "locales_libres"),
    ("que locales puedo alquilar", "locales_libres"),
    ("quisiera informacion para poner un negocio", "locales_libres"),
    ("cuanto cuesta el alquiler de un local", "locales_libres"),
    ("espacios vacios en el mall", "locales_libres"),

    # Buscar tienda o rubro
    ("donde esta la tienda", "buscar_tienda"),
    ("buscar una tienda en especifico", "buscar_tienda"),
    ("hay alguna farmacia o restaurante", "buscar_tienda"),
    ("que tiendas tienen disponibles", "buscar_tienda"),
    ("busco ropa comida tecnologia", "buscar_tienda"),
    ("en que piso queda el local", "buscar_tienda"),

    # Estado de incidencia
    ("estado de mi incidencia", "estado_incidencia"),
    ("como va mi reclamo o reporte", "estado_incidencia"),
    ("consultar el ticket inc", "estado_incidencia"),
    ("seguimiento a mi problema tecnico", "estado_incidencia"),
    ("ya revisaron mi incidencia", "estado_incidencia"),

    # Cupones y fidelización
    ("tengo cupones disponibles", "cupones"),
    ("que descuentos o promociones hay", "cupones"),
    ("como canjeo mis cupones", "cupones"),
    ("puntos de fidelizacion y ofertas", "cupones"),
]

FRASES = [normalizar_texto(f) for f, _ in DATASET]
ETIQUETAS = [e for _, e in DATASET]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(FRASES)

RESPUESTAS_NATURALES = {
    "saludo": [
        "¡Hola! Qué gusto saludarte. Soy el asistente virtual del Mall Center Caparazón. ¿En qué te puedo ayudar hoy? Puedes preguntarme por tiendas, horarios, ubicación o locales disponibles.",
        "¡Bienvenido a Caparazón! Estoy aquí para orientarte en tu visita. ¿Buscas alguna tienda en particular o necesitas información del mall?"
    ],
    "horario": [
        "Abrimos todos los días de lunes a domingo, desde las 09:00 hasta las 21:00 horas. ¡Te esperamos cuando gustes!",
        "Nuestro horario de atención al público es de 09:00 a 21:00 horas, de lunes a domingo (incluyendo fines de semana)."
    ],
    "ubicacion": [
        "Nos encontramos en la Zona La Vita, muy cerca al Puente Vita, en La Paz - Bolivia. Es una zona súper accesible.",
        "Estamos ubicados en la Zona La Vita (a pasos del Puente Vita), La Paz. Si vienes en transporte público o vehículo propio, el acceso es directo."
    ],
    "locales_libres": [
        "¡Excelente que quieras formar parte de Caparazón! Contamos con espacios comerciales disponibles en distintos niveles. Puedes consultar los locales libres en tiempo real o dejarnos tus datos en administración.",
        "Tenemos espacios disponibles ideales para distintos rubros. Permíteme verificar los locales libres actuales para darte el detalle."
    ],
    "buscar_tienda": [
        "Con gusto te ayudo a ubicar lo que buscas. También puedes explorar el Directorio de Tiendas aquí mismo en la página y hacer clic en cualquier negocio para ver su piso, teléfono y horario.",
        "Tenemos una gran variedad de comercios y servicios. Dime el nombre o rubro de la tienda que buscas, o revísala directamente en nuestro catálogo interactivo."
    ],
    "estado_incidencia": [
        "Entiendo, vamos a revisar el estado de tu reporte. Si tienes el código (por ejemplo INC-001), compártelo para darte el seguimiento exacto.",
        "Con gusto consulto cómo avanza tu incidencia. Por favor indícame el código de tu ticket o verifica en tu panel de locatario."
    ],
    "cupones": [
        "¡Tenemos excelentes beneficios para nuestros visitantes! Puedes revisar tus cupones activos y promociones vigentes en la sección de Fidelización.",
        "Nuestros locatarios constantemente lanzan descuentos especiales. Déjame revisar qué promociones y cupones tenemos activos para ti."
    ],
    "desconocido": [
        "Disculpa, no estoy seguro de haber entendido del todo tu consulta, pero quiero ayudarte. ¿Podrías comentarme si buscas información sobre tiendas, horarios, ubicación o alquiler de locales?",
        "¡Hola! Todavía estoy aprendiendo sobre algunas consultas específicas. Puedo ayudarte rápido con horarios, ubicación del mall, búsqueda de tiendas o disponibilidad de locales. ¿Qué necesitas?"
    ]
}

def extraer_entidades(texto_original: str, texto_limpio: str) -> dict:
    entidades = {}
    
    # Detectar código de incidencia (Ej. INC-001 o INC-2026)
    match_inc = re.search(r"\binc[-\s]?\d+\b", texto_original, re.IGNORECASE)
    if match_inc:
        entidades["codigo_incidencia"] = match_inc.group(0).upper().replace(" ", "-")

    # Detectar código de espacio/local (Ej. L-101, P1-02)
    match_loc = re.search(r"\b[lp]\d?[-\s]?\d{2,3}\b", texto_original, re.IGNORECASE)
    if match_loc:
        entidades["codigo_local"] = match_loc.group(0).upper().replace(" ", "-")

    # Detectar rubros comunes mencionados
    rubros = ["farmacia", "ropa", "comida", "restaurante", "cafeteria", "tecnologia", "celular", "zapateria", "banco", "peluqueria"]
    for rubro in rubros:
        if rubro in texto_limpio:
            entidades["rubro"] = rubro
            break

    return entidades

@app.post("/nlp")
def procesar(mensaje: Mensaje):
    texto_limpio = normalizar_texto(mensaje.mensaje)
    user_vec = vectorizer.transform([texto_limpio])
    similitudes = cosine_similarity(user_vec, X).flatten()
    mejor_idx = similitudes.argmax()
    confianza = float(similitudes[mejor_idx])

    entidades = extraer_entidades(mensaje.mensaje, texto_limpio)

    if confianza < 0.28:
        intencion = "desconocido"
    else:
        intencion = ETIQUETAS[mejor_idx]

    respuesta = random.choice(RESPUESTAS_NATURALES[intencion])

    return {
        "intencion": intencion,
        "confianza": round(confianza, 4),
        "entidades": entidades,
        "respuesta_sugerida": respuesta
    }

@app.get("/estado")
def estado_servicio():
    return {"servicio": "Caparazon IA", "estado": "activo"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)