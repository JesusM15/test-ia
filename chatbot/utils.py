import spacy
import unicodedata

nlp = spacy.load("es_core_news_sm")

def remove_accents(text: str):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def detect_intent_spacy(pregunta: str):
    doc = nlp(remove_accents(pregunta.lower()))
    
    tokens = [remove_accents(token.lemma_.lower()) for token in doc] + \
             [remove_accents(token.text.lower()) for token in doc]

    if "comprador" in tokens and ("mejor" in tokens or "mas" in tokens):
        return "top_compradores"
    elif "deudor" in tokens and ("alto" in tokens or "mas" in tokens):
        return "top_deudores"
    elif ("cuanto" in tokens) and "comprador" in tokens:
        return "contar_compradores"
    else:
        return "no_entendido"
