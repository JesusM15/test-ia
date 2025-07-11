** VERSION DE PYTHON UTILIZADA ** 
python 3.11.5, en cuanto las librerias utilizadas son las siguientes:

fastapi
uvicorn
pillow
pytesseract
opencv-python
python-multipart
sqlalchemy
requests
beautifulsoup4
spacy
es_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.5.0/es_core_news_sm-3.5.0.tar.gz
torchvision
torch
numpy


# Instrucciones Generales

Este proyecto está dividido en cinco partes, cada una ubicada en su propia carpeta con un `requirements.txt` correspondiente. Para ejecutar cualquier parte, sigue estos pasos:

# 1. Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt

# 2. Ejecutar proyectos
Antes de esto es necesario para cada proyecto en cada carpeta acceder a su carpeta de entorno virtuaal y ejecutar el pip install -r requirements.txt

1.- OCR 

Para este proyecto es necesario ejecutar el siguiente comando:
uvicorn main:app --reload

situado en la carpeta ocr-web-fullstack, para ejecutar la parte del frontend es necesario un 
se uso Node 18.18.0v
npm install 
npm run dev

de esta forma podras utilizar sin problemas el backend

2.- El resto de proyectos solo necesitan ejecutar el api de esta forma:
uvicorn main:app --reload


# Desafios 

Dividire por secciones los principales desafios a los que me he enfrentado en este proyecto

1.- En el ocr fue que no se interpretaban correctamente algunos datos, para ello decidi aplicar una conversion de imagen a color a blanco y negro para marcar más las letras del fondo así como aplicar un blur.

2.- En este no encuentro muchos desafios o una decisión clave, lo principal fue poder tener la base de datos, crear el modelo y una vez eso simplemente se explora la página para extraer los datos posteriormente se crean los endpoints en los datos

3.- Para el chatbot fue complicado entrenarlo, decidi principalmente basarme en 4 respuestas generales que daria el API a ciertos token o claves de palabras para que de esta forma no hubiese mucho fallo y pueda funcionar con menos cantidad de datos.


#endpoinst documentados

=====================================
PARTE 1 – OCR en Web App
=====================================

[POST] /upload-ocr
Descripción: Recibe una imagen y devuelve el texto detectado con OCR.
Tipo de contenido: multipart/form-data
Campo esperado: file (imagen)

Ejemplo con Postman:
- Método: POST
- URL: http://localhost:8000/upload-ocr
- Body → form-data:
    file = imagen.png

Respuesta esperada:
{
  "text": "Total: $123.45\nFecha: 12/03/2024"
}


=====================================
PARTE 2 – Web Scraping + Filtros API
=====================================

[GET] /products
Descripción: Devuelve productos guardados, con soporte para filtros por nombre, precio y categoría.

Parámetros opcionales:
- name (str)
- category (str)
- min_price (float)
- max_price (float)

Ejemplo de uso:
http://localhost:8000/products?category=Books&min_price=10&max_price=30&name=python

Respuesta esperada:
[
  {
    "id": 1,
    "title": "Python para todos",
    "price": 22.5,
    "category": "Books",
    "rating": 4
  }
]


=====================================
PARTE 3 – Chatbot + NLP
=====================================

[GET] /chatbot
Descripción: Recibe una pregunta en lenguaje natural y responde con base en los datos de compradores y deudores.

Parámetro requerido:
- pregunta (str)

Ejemplo:
http://localhost:8000/chatbot?pregunta=¿Quiénes son los mejores compradores?

Respuesta esperada:
{
  "respuesta": "🛒 Los mejores compradores son:\n- Marta Gómez: $2100.75\n- Ana Torres: $1800.50\n- Luis Pérez: $950.00"
}


=====================================
PARTE 4 – CNN + API Predictiva
=====================================

[POST] /predict-image
Descripción: Recibe una imagen en formato base64 y devuelve la predicción del número manuscrito (0–9).

Tipo de contenido: application/json

Formato JSON esperado:
{
  "image_base64": "<cadena_base64>"
}

Respuesta esperada:
{
  "prediction": 7
}

Ejemplo de prueba con Postman:
- Método: POST
- URL: http://localhost:8000/predict-image
- Body → raw → JSON:
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
