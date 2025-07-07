from fastapi import FastAPI, Query
from database import engine, SessionLocal
from models import Base, Product
from typing import List, Optional

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/products")
def get_products(
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
    name: Optional[str] = None
):
    db = SessionLocal()
    query = db.query(Product)

    if min_price != None:
        query = query.filter(Product.price >= min_price)
    if max_price != None:
        query = query.filter(Product.price <= max_price)
    if category:
        query = query.filter(Product.category.ilike(f"%{category}%"))
    if name:
        query = query.filter(Product.title.ilike(f"%{name}%"))

    results = query.all()
    db.close()
    return results