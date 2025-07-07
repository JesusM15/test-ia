from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io
import cv2 as cv
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/upload-ocr")
async def upload_ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparray = np.frombuffer(contents, np.uint8)
        fullColorImage = cv.imdecode(nparray, cv.IMREAD_COLOR)

        grayImg = cv.cvtColor(fullColorImage, cv.COLOR_BGR2GRAY)
        grayImg = cv.medianBlur(grayImg, 3)


        image = Image.fromarray(grayImg)
        custom_config = r'--oem 3 --psm 6'
        textFromImage = pytesseract.image_to_string(image, config=custom_config).strip()

        if not textFromImage: raise HTTPException(status_code=400, detail=f"No hay texto en la imagen")

        return { "text": textFromImage }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error al subir imagen: {error}")