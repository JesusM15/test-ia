from sqlalchemy import Column, Integer, String, Float
from database import Base

class Comprador(Base):

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True)
    total_compras = Column(Float)

    __tablename__ = "compradores"

class Deudor(Base):
    id = Column(Integer, primary_key=True, index = True)
    nombre = Column(String, index=True)
    monto_adeudado= Column(Float)

    __tablename__ = "deudores"