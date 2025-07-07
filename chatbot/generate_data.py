from database import SessionLocal
from models import Comprador, Deudor
from sqlalchemy import delete

db = SessionLocal()

db.execute(delete(Comprador))
db.execute(delete(Deudor))
db.commit()

compradores = [
    Comprador(nombre="Ana Torres", total_compras=1800.50),
    Comprador(nombre="Luis Pérez", total_compras=950.00),
    Comprador(nombre="Marta Gómez", total_compras=2100.75),
    Comprador(nombre="Javier Luna", total_compras=150.00),
]

deudores = [
    Deudor(nombre="Pedro Ramírez", monto_adeudado=800.00),
    Deudor(nombre="Carmen López", monto_adeudado=1200.25),
    Deudor(nombre="Raúl Martínez", monto_adeudado=500.00),
    Deudor(nombre="Lucía Vega", monto_adeudado=50.00),
]

db.add_all(compradores)
db.add_all(deudores)
db.commit()
db.close()
