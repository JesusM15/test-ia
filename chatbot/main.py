from fastapi import FastAPI, Query
from models import Comprador, Deudor
from database import SessionLocal, engine
from models import Base
from utils import detect_intent_spacy

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/chatbot")
def chatbot(pregunta: str = Query(..., description="Pregunta realizada al chatbot")):
    db = SessionLocal()
    intent = detect_intent_spacy(pregunta)

    if intent == "top_compradores":
        compradores = db.query(Comprador).order_by(Comprador.total_compras.desc()).limit(3).all()
        response = "Los mejores compradores son:\n" + "\n".join([f"- {c.nombre}: ${c.total_compras:.2f}" for c in compradores])

    elif intent == "top_deudores":
        deudores = db.query(Deudor).order_by(Deudor.monto_adeudado.desc()).limit(3).all()
        response = "Los deudores más altos son:\n" + "\n".join([f"- {d.nombre}: ${d.monto_adeudado:.2f}" for d in deudores])

    elif intent == "contar_compradores":
        total = db.query(Comprador).count()
        response = f"Actualmente hay {total} compradores registrados."
    else:
        response = "Lo siento, no he comprendido tu pregunta."

    db.close()
    return {"respuesta": response}
