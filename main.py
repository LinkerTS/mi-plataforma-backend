from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sqlite3

app = FastAPI()

# ---------- BASE DE DATOS ----------
def init_db():
    conn = sqlite3.connect("activos.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activos (
            id INTEGER PRIMARY KEY,
            nombre TEXT,
            descripcion TEXT,
            categoria TEXT,
            ubicacion TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------- MODELOS ----------
class Asset(BaseModel):
    id: int
    nombre: str
    descripcion: str
    categoria: str
    ubicacion: str

class AssetUpdate(BaseModel):
    nombre: Optional[str]
    descripcion: Optional[str]
    categoria: Optional[str]
    ubicacion: Optional[str]

# ---------- ENDPOINTS ----------
@app.get("/")
def root():
    return {"message": "API de gestión de activos funcionando"}

@app.get("/activos")
def get_assets() -> List[Asset]:
    conn = sqlite3.connect("activos.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, descripcion, categoria, ubicacion FROM activos")
    rows = cursor.fetchall()
    conn.close()
    return [Asset(id=row[0], nombre=row[1], descripcion=row[2], categoria=row[3], ubicacion=row[4]) for row in rows]

@app.post("/activos")
def create_asset(asset: Asset):
    conn = sqlite3.connect("activos.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM activos WHERE id = ?", (asset.id,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="El ID ya existe")
    cursor.execute("INSERT INTO activos (id, nombre, descripcion, categoria, ubicacion) VALUES (?, ?, ?, ?, ?)",
                   (asset.id, asset.nombre, asset.descripcion, asset.categoria, asset.ubicacion))
    conn.commit()
    conn.close()
    return {"message": "Activo agregado exitosamente", "data": asset}

@app.get("/activos/{asset_id}")
def get_asset_by_id(asset_id: int):
    conn = sqlite3.connect("activos.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, descripcion, categoria, ubicacion FROM activos WHERE id = ?", (asset_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return Asset(id=row[0], nombre=row[1], descripcion=row[2], categoria=row[3], ubicacion=row[4])
    else:
        raise HTTPException(status_code=404, detail="Activo no encontrado")

@app.put("/activos/{asset_id}")
def update_asset(asset_id: int, asset_update: AssetUpdate):
    conn = sqlite3.connect("activos.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM activos WHERE id = ?", (asset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Activo no encontrado")

    if asset_update.nombre:
        cursor.execute("UPDATE activos SET nombre = ? WHERE id = ?", (asset_update.nombre, asset_id))
    if asset_update.descripcion:
        cursor.execute("UPDATE activos SET descripcion = ? WHERE id = ?", (asset_update.descripcion, asset_id))
    if asset_update.categoria:
        cursor.execute("UPDATE activos SET categoria = ? WHERE id = ?", (asset_update.categoria, asset_id))
    if asset_update.ubicacion:
        cursor.execute("UPDATE activos SET ubicacion = ? WHERE id = ?", (asset_update.ubicacion, asset_id))

    conn.commit()
    conn.close()
    return {"message": "Activo actualizado correctamente"}

@app.delete("/activos/{asset_id}")
def delete_asset(asset_id: int):
    conn = sqlite3.connect("activos.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM activos WHERE id = ?", (asset_id,))
    conn.commit()
    conn.close()
    return {"message": "Activo eliminado correctamente"}
