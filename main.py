from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import os, sqlite3, io

# =========================
# APP BASE
# =========================
app = FastAPI(title="API Gestión de Activos", version="1.1.0")

# CORS (para pruebas deja "*"; en prod pon tu dominio de Lovable)
ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DB
# =========================
DB_NAME = "activos.db"

def conn():
    return sqlite3.connect(DB_NAME)

def init_db():
    c = conn(); cur = c.cursor()
    # Tabla de activos
    cur.execute("""
    CREATE TABLE IF NOT EXISTS activos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        descripcion TEXT,
        categoria TEXT,
        ubicacion TEXT,
        factura TEXT,
        proveedor TEXT,
        tipo TEXT,
        area_responsable TEXT,
        fecha_ingreso TEXT,
        valor_adquisicion REAL,
        estado TEXT,
        creado_en TEXT
    )
    """)
    # Tabla de usuarios (auth)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)
    # Tabla de etiquetas ML (para entrenar luego)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ml_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id INTEGER NOT NULL,
        label INTEGER NOT NULL,        -- 1: crítico/obsoleto, 0: ok
        timestamp TEXT NOT NULL,
        FOREIGN KEY(asset_id) REFERENCES activos(id)
    )
    """)
    c.commit(); c.close()

init_db()

# =========================
# MODELOS
# =========================
class AssetCreate(BaseModel):
    nombre: str
    descripcion: Optional[str] = ""
    categoria: Optional[str] = ""
    ubicacion: Optional[str] = ""
    factura: Optional[str] = ""
    proveedor: Optional[str] = ""
    tipo: Optional[str] = ""
    area_responsable: Optional[str] = ""
    fecha_ingreso: Optional[str] = ""  # "YYYY-MM-DD"
    valor_adquisicion: Optional[float] = None
    estado: Optional[str] = "activo"

class AssetOut(BaseModel):
    id: int
    nombre: str
    descripcion: Optional[str]
    categoria: Optional[str]
    ubicacion: Optional[str]
    factura: Optional[str]
    proveedor: Optional[str]
    tipo: Optional[str]
    area_responsable: Optional[str]
    fecha_ingreso: Optional[str]
    valor_adquisicion: Optional[float]
    estado: Optional[str]
    creado_en: str

class AssetUpdate(BaseModel):
    nombre: Optional[str]
    descripcion: Optional[str]
    categoria: Optional[str]
    ubicacion: Optional[str]
    factura: Optional[str]
    proveedor: Optional[str]
    tipo: Optional[str]
    area_responsable: Optional[str]
    fecha_ingreso: Optional[str]
    valor_adquisicion: Optional[float]
    estado: Optional[str]

class UserCreate(BaseModel):
    email: str
    password: str
    role: Optional[str] = None  # si no se envía, asignamos por dominio

class BajaRequest(BaseModel):
    motivo: str

# =========================
# AUTH + ROLES (JWT)
# =========================
from passlib.hash import bcrypt
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "devsecret-change-me")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Mapea dominios de correo -> rol por defecto
DOMAIN_ROLE_MAP = {
    "tecnica.com": "technical",
    "bodega.com": "storekeeper",
    "gerencia.com": "management",
    "supervision.com": "supervisor"
}

def ensure_admin_seed():
    c = conn(); cur = c.cursor()
    cur.execute("SELECT id FROM users WHERE email=?", ("admin@demo.com",))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (email,password_hash,role) VALUES (?,?,?)",
                    ("admin@demo.com", bcrypt.hash("admin123"), "management"))
        c.commit()
    c.close()

ensure_admin_seed()

def create_token(email: str, role: str):
    payload = {
        "sub": email,
        "role": role,
        "exp": (datetime.utcnow() + timedelta(hours=8))
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    email = payload.get("sub"); role = payload.get("role")
    if not email:
        raise HTTPException(status_code=401, detail="Token inválido")
    c = conn(); cur = c.cursor()
    cur.execute("SELECT email, role FROM users WHERE email=?", (email,))
    row = cur.fetchone(); c.close()
    if not row:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    return {"email": row[0], "role": row[1]}

def require_role(roles: List[str]):
    def wrapper(user = Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Permisos insuficientes")
        return user
    return wrapper

@app.post("/auth/register")
def register(user: UserCreate, current=Depends(require_role(["management"]))):
    # Asignación automática por dominio si no envía role
    role = (user.role or "").strip()
    if not role:
        try:
            domain = user.email.split("@")[1].lower()
            role = DOMAIN_ROLE_MAP.get(domain, "storekeeper")
        except:
            role = "storekeeper"
    c = conn(); cur = c.cursor()
    try:
        cur.execute("INSERT INTO users (email,password_hash,role) VALUES (?,?,?)",
                    (user.email, bcrypt.hash(user.password), role))
        c.commit()
    except sqlite3.IntegrityError:
        c.close(); raise HTTPException(status_code=400, detail="Email ya registrado")
    c.close()
    return {"message": "Usuario creado", "role_asignado": role}

@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    c = conn(); cur = c.cursor()
    cur.execute("SELECT email, password_hash, role FROM users WHERE email=?", (form.username,))
    row = cur.fetchone(); c.close()
    if not row or not bcrypt.verify(form.password, row[1]):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    token = create_token(row[0], row[2])
    return {"access_token": token, "token_type": "bearer", "role": row[2]}

# =========================
# ENDPOINTS CRUD
# =========================
@app.get("/")
def root():
    return {"message": "API de gestión de activos funcionando"}

@app.get("/activos", response_model=List[AssetOut])
def get_assets(user=Depends(get_current_user)):
    c = conn(); cur = c.cursor()
    cur.execute("""SELECT id, nombre, descripcion, categoria, ubicacion, factura, proveedor,
                          tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en
                   FROM activos ORDER BY id DESC""")
    rows = cur.fetchall(); c.close()
    return [AssetOut(
        id=r[0], nombre=r[1], descripcion=r[2], categoria=r[3], ubicacion=r[4],
        factura=r[5], proveedor=r[6], tipo=r[7], area_responsable=r[8],
        fecha_ingreso=r[9], valor_adquisicion=r[10], estado=r[11], creado_en=r[12]
    ) for r in rows]

@app.get("/activos/{asset_id}", response_model=AssetOut)
def get_asset(asset_id: int, user=Depends(get_current_user)):
    c = conn(); cur = c.cursor()
    cur.execute("""SELECT id, nombre, descripcion, categoria, ubicacion, factura, proveedor,
                          tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en
                   FROM activos WHERE id=?""", (asset_id,))
    r = cur.fetchone(); c.close()
    if not r:
        raise HTTPException(status_code=404, detail="Activo no encontrado")
    return AssetOut(
        id=r[0], nombre=r[1], descripcion=r[2], categoria=r[3], ubicacion=r[4],
        factura=r[5], proveedor=r[6], tipo=r[7], area_responsable=r[8],
        fecha_ingreso=r[9], valor_adquisicion=r[10], estado=r[11], creado_en=r[12]
    )

@app.post("/activos", response_model=AssetOut, status_code=201)
def create_asset(asset: AssetCreate, user=Depends(require_role(["storekeeper","supervisor"]))):
    c = conn(); cur = c.cursor()
    creado_en = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO activos (nombre, descripcion, categoria, ubicacion, factura, proveedor, tipo,
                             area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (asset.nombre, asset.descripcion, asset.categoria, asset.ubicacion, asset.factura, asset.proveedor,
          asset.tipo, asset.area_responsable, asset.fecha_ingreso, asset.valor_adquisicion, asset.estado, creado_en))
    c.commit()
    new_id = cur.lastrowid
    c.close()
    return get_asset(new_id, user)

@app.put("/activos/{asset_id}")
def update_asset(asset_id: int, up: AssetUpdate, user=Depends(require_role(["supervisor","technical"]))):
    c = conn(); cur = c.cursor()
    cur.execute("SELECT id FROM activos WHERE id=?", (asset_id,))
    if not cur.fetchone():
        c.close(); raise HTTPException(status_code=404, detail="Activo no encontrado")
    fields, values = [], []
    for k, v in up.dict(exclude_unset=True).items():
        fields.append(f"{k}=?"); values.append(v)
    if fields:
        q = f"UPDATE activos SET {', '.join(fields)} WHERE id=?"
        values.append(asset_id)
        cur.execute(q, tuple(values)); c.commit()
    c.close()
    return {"message": "Activo actualizado"}

@app.delete("/activos/{asset_id}")
def delete_asset(asset_id: int, user=Depends(require_role(["management"]))):
    c = conn(); cur = c.cursor()
    cur.execute("DELETE FROM activos WHERE id=?", (asset_id,))
    c.commit(); c.close()
    return {"message": "Activo eliminado"}

# =========================
# EXPORT PARA POWER BI (DETALLE)
# =========================
@app.get("/export/activos.csv")
def export_csv(user=Depends(get_current_user)):
    import csv
    c = conn(); cur = c.cursor()
    cur.execute("""SELECT id, nombre, categoria, ubicacion, tipo, area_responsable,
                          fecha_ingreso, valor_adquisicion, estado, creado_en
                   FROM activos ORDER BY id DESC""")
    rows = cur.fetchall(); c.close()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id","nombre","categoria","ubicacion","tipo","area_responsable",
                     "fecha_ingreso","valor_adquisicion","estado","creado_en"])
    writer.writerows(rows)
    return buffer.getvalue()

@app.get("/export/activos.json")
def export_json(user=Depends(get_current_user)):
    c = conn(); cur = c.cursor()
    cur.execute("""SELECT id, nombre, categoria, ubicacion, tipo, area_responsable,
                          fecha_ingreso, valor_adquisicion, estado, creado_en
                   FROM activos ORDER BY id DESC""")
    rows = cur.fetchall(); c.close()
    cols = ["id","nombre","categoria","ubicacion","tipo","area_responsable",
            "fecha_ingreso","valor_adquisicion","estado","creado_en"]
    data = [dict(zip(cols, r)) for r in rows]
    return {"data": data}

# =========================
# ENDPOINT OPTIMIZADO PARA POWER BI (KPIs + DETALLE)
# =========================
@app.get("/reportes/activos")
def reportes_activos(user=Depends(get_current_user)):
    """
    Devuelve KPIs + tablas agregadas + detalle.
    Power BI puede conectarse aquí y usar ambas cosas.
    """
    c = conn(); cur = c.cursor()
    cur.execute("""SELECT id, nombre, categoria, ubicacion, tipo, area_responsable,
                          fecha_ingreso, valor_adquisicion, estado, creado_en
                   FROM activos""")
    rows = cur.fetchall()
    c.close()

    cols = ["id","nombre","categoria","ubicacion","tipo","area_responsable",
            "fecha_ingreso","valor_adquisicion","estado","creado_en"]
    data = [dict(zip(cols, r)) for r in rows]

    # KPIs básicos
    total = len(data)
    activos = sum(1 for d in data if (d["estado"] or "").lower() == "activo")
    bajas = sum(1 for d in data if (d["estado"] or "").lower() == "baja")

    # Por estado
    por_estado = {}
    for d in data:
        key = (d["estado"] or "desconocido").lower()
        por_estado[key] = por_estado.get(key, 0) + 1

    # Por categoría
    por_categoria = {}
    for d in data:
        key = (d["categoria"] or "sin_categoria")
        por_categoria[key] = por_categoria.get(key, 0) + 1

    # Por mes de ingreso (YYYY-MM)
    por_mes = {}
    for d in data:
        fi = (d["fecha_ingreso"] or "")
        ym = fi[:7] if len(fi) >= 7 else "sin_fecha"
        por_mes[ym] = por_mes.get(ym, 0) + 1

    return {
        "kpis": {
            "total_activos": total,
            "activos_operacionales": activos,
            "activos_baja": bajas,
        },
        "agregados": {
            "por_estado": por_estado,
            "por_categoria": por_categoria,
            "por_mes_ingreso": por_mes
        },
        "detalle": data
    }

# =========================
# DOCUMENTOS PDF (COMPROBANTE INGRESO, ACTA DE BAJA, REPORTE EJECUTIVO)
# =========================
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def _fetch_asset(asset_id: int):
    c = conn(); cur = c.cursor()
    cur.execute("""SELECT id, nombre, descripcion, categoria, ubicacion, factura, proveedor,
                          tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en
                   FROM activos WHERE id=?""", (asset_id,))
    r = cur.fetchone(); c.close()
    if not r:
        raise HTTPException(status_code=404, detail="Activo no encontrado")
    cols = ["id","nombre","descripcion","categoria","ubicacion","factura","proveedor",
            "tipo","area_responsable","fecha_ingreso","valor_adquisicion","estado","creado_en"]
    return dict(zip(cols, r))

def _pdf_from_pairs(title: str, pairs: list):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title)
    y -= 25
    c.setFont("Helvetica", 10)
    for k, v in pairs:
        if y < 80:
            c.showPage(); y = h - 50; c.setFont("Helvetica", 10)
        c.drawString(50, y, f"{k}: {'' if v is None else v}")
        y -= 18
    c.showPage(); c.save()
    buf.seek(0)
    return buf

@app.get("/documentos/comprobante/ingreso/{asset_id}.pdf")
def comprobante_ingreso(asset_id: int, user=Depends(get_current_user)):
    a = _fetch_asset(asset_id)
    hoy = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    pairs = [
        ("COMPROBANTE DE INGRESO", ""),
        ("Fecha de emisión", hoy),
        ("ID", a["id"]),
        ("Nombre", a["nombre"]),
        ("Descripción", a.get("descripcion","")),
        ("Categoría", a.get("categoria","")),
        ("Tipo", a.get("tipo","")),
        ("Ubicación", a.get("ubicacion","")),
        ("Área responsable", a.get("area_responsable","")),
        ("Proveedor", a.get("proveedor","")),
        ("Factura", a.get("factura","")),
        ("Fecha ingreso", a.get("fecha_ingreso","")),
        ("Valor adquisición", a.get("valor_adquisicion","")),
        ("Estado", a.get("estado","")),
        ("Registrado en sistema", a.get("creado_en","")),
    ]
    buf = _pdf_from_pairs("Comprobante de Ingreso de Activo", pairs)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="comprobante_ingreso_{asset_id}.pdf"'})

@app.post("/activos/{asset_id}/baja")
def dar_baja(asset_id:int, req: BajaRequest, user=Depends(require_role(["supervisor","management"]))):
    c = conn(); cur = c.cursor()
    cur.execute("UPDATE activos SET estado=? WHERE id=?", ("baja", asset_id))
    c.commit(); c.close()
    a = _fetch_asset(asset_id)
    hoy = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    pairs = [
        ("ACTA DE BAJA", ""),
        ("Fecha", hoy),
        ("ID", a["id"]),
        ("Nombre", a["nombre"]),
        ("Motivo de baja", req.motivo),
        ("Ubicación", a.get("ubicacion","")),
        ("Área responsable", a.get("area_responsable","")),
        ("Valor adquisición", a.get("valor_adquisicion","")),
        ("Estado posterior", "baja"),
    ]
    buf = _pdf_from_pairs("Acta de Baja de Bien", pairs)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="acta_baja_${asset_id}.pdf"'})

@app.get("/reportes/instantaneo.pdf")
def reporte_ejecutivo_pdf(user=Depends(get_current_user)):
    """
    Genera un PDF ejecutivo con KPIs y cortes principales (conteos).
    NO es el PDF de Power BI (ese requiere licencias). Es un reporte local
    para descargar automáticamente desde la plataforma.
    """
    rep = reportes_activos(user)  # reutiliza el endpoint optimizado
    k = rep["kpis"]
    agg = rep["agregados"]

    hoy = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        ("REPORTE EJECUTIVO - ACTIVOS", ""),
        ("Fecha de emisión", hoy),
        ("---- KPIs ----", ""),
        ("Total de activos", k["total_activos"]),
        ("Activos operacionales", k["activos_operacionales"]),
        ("Activos de baja", k["activos_baja"]),
        ("", ""),
        ("---- Por estado ----", ""),
    ]
    for est, cnt in sorted(agg["por_estado"].items()):
        lines.append((f"  {est}", cnt))
    lines.append(("", ""))
    lines.append(("---- Por categoría ----", ""))
    for cat, cnt in sorted(agg["por_categoria"].items()):
        lines.append((f"  {cat}", cnt))
    lines.append(("", ""))
    lines.append(("---- Ingresos por mes (YYYY-MM) ----", ""))
    for mes, cnt in sorted(agg["por_mes_ingreso"].items()):
        lines.append((f"  {mes}", cnt))

    buf = _pdf_from_pairs("Reporte Ejecutivo de Activos", lines)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="reporte_ejecutivo.pdf"'})
# =========================
# FIN
# =========================
