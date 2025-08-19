from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import os, sqlite3, io
import csv, json, math
from io import TextIOWrapper
from typing import Iterable
import joblib  # <-- NUEVO: para guardar/cargar el modelo a archivo

# =========================
# APP BASE
# =========================
APP_VERSION = "1.3.4"
app = FastAPI(title="API Gestión de Activos", version=APP_VERSION)

# CORS desde variables de entorno (por defecto "*")
origins_env = os.getenv("ORIGINS", "*")
ORIGINS = [""] if origins_env.strip() == "" else [o.strip() for o in origins_env.split(",") if o.strip()]
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
DB_NAME = os.getenv("DB_NAME", "activos.db")

def get_conn():
    # SQLite seguro para uso en servicio web (autocommit + WAL)
    conn = sqlite3.connect(DB_NAME, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    with get_conn() as c:
        cur = c.cursor()
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

# --- Migración automática: agrega columnas faltantes en 'activos' si la DB es vieja ---
def ensure_activos_columns():
    """
    Migra la tabla 'activos' si faltan columnas de versiones anteriores.
    No borra datos; solo agrega columnas faltantes con valores NULL.
    """
    needed = {
        "descripcion": "TEXT",
        "categoria": "TEXT",
        "ubicacion": "TEXT",
        "factura": "TEXT",
        "proveedor": "TEXT",
        "tipo": "TEXT",
        "area_responsable": "TEXT",
        "fecha_ingreso": "TEXT",
        "valor_adquisicion": "REAL",
        "estado": "TEXT",
        "creado_en": "TEXT",
    }
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("PRAGMA table_info(activos)")
        existing = {row[1] for row in cur.fetchall()}
        for col, coltype in needed.items():
            if col not in existing:
                cur.execute(f"ALTER TABLE activos ADD COLUMN {col} {coltype}")

# Inicializa y migra
init_db()
ensure_activos_columns()

# ===== BLOQUE A — IA SETTINGS (guardar modelo/pesos/umbral) =====
def ensure_settings_table():
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)

def _get_setting(key: str, default=None):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return row[0]

def _set_setting(key: str, value):
    val = json.dumps(value)
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, val))

# Crear tabla settings
ensure_settings_table()

# ===== Persistencia a archivo del modelo (NUEVO) =====
MODEL_PATH = os.getenv("MODEL_PATH", "modelo.pkl")

def _save_model_file(weights: list, features: list, threshold: float):
    """
    Guarda una copia del modelo en archivo. La DB 'settings' sigue siendo la fuente de verdad.
    """
    payload = {
        "weights": weights,       # incluye bias en weights[0]
        "features": features,     # orden de features (sin bias)
        "threshold": float(threshold),
        "version": APP_VERSION,
        "saved_at": datetime.utcnow().isoformat(),
        "algo": "logreg-gd",
    }
    try:
        joblib.dump(payload, MODEL_PATH)
        return True
    except Exception as e:
        print("WARN: no se pudo guardar modelo.pkl:", repr(e))
        return False

def _load_model_file():
    """
    Carga el modelo desde archivo si existe. Devuelve (weights, features, threshold) o (None,None,None).
    """
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None, None
        payload = joblib.load(MODEL_PATH)
        return payload.get("weights"), payload.get("features"), float(payload.get("threshold", 0.5))
    except Exception as e:
        print("WARN: no se pudo cargar modelo.pkl:", repr(e))
        return None, None, None

def _sync_model_persistence():
    """
    Sincroniza DB <-> archivo al iniciar:
      - Si hay archivo y DB vacía -> importar a DB.
      - Si hay DB y no hay archivo -> crear archivo.
      - Si hay ambos -> no-op (DB es la fuente de verdad).
    """
    w_db = _get_setting("AI_LOGREG_WEIGHTS", None)
    f_db = _get_setting("AI_FEATURE_NAMES", None)
    t_db = _get_setting("AI_THRESHOLD_CRITICAL", None)

    w_f, f_f, t_f = _load_model_file()

    # Caso 1: archivo existe y DB está vacía -> importar a DB
    if (w_f and f_f) and not (w_db and f_db):
        _set_setting("AI_LOGREG_WEIGHTS", w_f)
        _set_setting("AI_FEATURE_NAMES", f_f)
        _set_setting("AI_THRESHOLD_CRITICAL", float(t_f if t_f is not None else 0.5))
        print("SYNC: importado modelo desde archivo -> DB")
        return

    # Caso 2: DB existe y archivo falta -> exportar a archivo
    if (w_db and f_db) and not (w_f and f_f):
        _save_model_file(w_db, f_db, float(t_db if t_db is not None else 0.5))
        print("SYNC: exportado modelo desde DB -> archivo")
        return

    # Caso 3: ambos presentes -> no-op
    print("SYNC: modelo presente en DB (y archivo opcional)")

@app.on_event("startup")
def _startup_model_sync():
    try:
        _sync_model_persistence()
    except Exception as e:
        print("WARN startup sync:", repr(e))

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
# AUTH + ROLES (JWT)  -> PBKDF2 en lugar de bcrypt
# =========================
from passlib.hash import pbkdf2_sha256 as hasher
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
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT id FROM users WHERE email=?", ("admin@demo.com",))
        if not cur.fetchone():
            cur.execute("INSERT INTO users (email,password_hash,role) VALUES (?,?,?)",
                        ("admin@demo.com", hasher.hash("admin123"), "management"))
ensure_admin_seed()

def create_token(email: str, role: str):
    payload = {"sub": email, "role": role, "exp": (datetime.utcnow() + timedelta(hours=8))}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    email = payload.get("sub"); role = payload.get("role")
    if not email:
        raise HTTPException(status_code=401, detail="Token inválido")
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT email, role FROM users WHERE email=?", (email,))
        row = cur.fetchone()
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
    role = (user.role or "").strip()
    if not role:
        try:
            domain = user.email.split("@")[1].lower()
            role = DOMAIN_ROLE_MAP.get(domain, "storekeeper")
        except:
            role = "storekeeper"
    try:
        with get_conn() as c:
            cur = c.cursor()
            cur.execute("INSERT INTO users (email,password_hash,role) VALUES (?,?,?)",
                        (user.email, hasher.hash(user.password), role))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email ya registrado")
    return {"message": "Usuario creado", "role_asignado": role}

@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT email, password_hash, role FROM users WHERE email=?", (form.username,))
        row = cur.fetchone()
        if not row or not hasher.verify(form.password, row[1]):
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
    token = create_token(row[0], row[2])
    return {"access_token": token, "token_type": "bearer", "role": row[2]}

# =========================
# HELPERS EXPORT
# =========================
EXPORT_COLUMNS = ["id","nombre","categoria","ubicacion","tipo","area_responsable",
                  "fecha_ingreso","valor_adquisicion","estado","creado_en"]

def fetch_export_rows():
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(f"""SELECT {", ".join(EXPORT_COLUMNS)}
                        FROM activos ORDER BY id DESC""")
        return cur.fetchall()

# =========================
# ENDPOINTS CRUD
# =========================
@app.get("/")
def root():
    return {"message": "API de gestión de activos funcionando", "version": APP_VERSION}

@app.get("/activos", response_model=List[AssetOut])
def get_assets(user=Depends(get_current_user)):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT id, nombre, descripcion, categoria, ubicacion, factura, proveedor,
                              tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en
                       FROM activos ORDER BY id DESC""")
        rows = cur.fetchall()
    return [AssetOut(
        id=r[0], nombre=r[1], descripcion=r[2], categoria=r[3], ubicacion=r[4],
        factura=r[5], proveedor=r[6], tipo=r[7], area_responsable=r[8],
        fecha_ingreso=r[9], valor_adquisicion=r[10], estado=r[11], creado_en=r[12]
    ) for r in rows]

@app.get("/activos/{asset_id}", response_model=AssetOut)
def get_asset(asset_id: int, user=Depends(get_current_user)):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT id, nombre, descripcion, categoria, ubicacion, factura, proveedor,
                              tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en
                       FROM activos WHERE id=?""", (asset_id,))
        r = cur.fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Activo no encontrado")
    return AssetOut(
        id=r[0], nombre=r[1], descripcion=r[2], categoria=r[3], ubicacion=r[4],
        factura=r[5], proveedor=r[6], tipo=r[7], area_responsable=r[8],
        fecha_ingreso=r[9], valor_adquisicion=r[10], estado=r[11], creado_en=r[12]
    )

@app.post("/activos", response_model=AssetOut, status_code=201)
def create_asset(asset: AssetCreate, user=Depends(require_role(["storekeeper","supervisor"]))):
    creado_en = datetime.utcnow().isoformat()
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""
            INSERT INTO activos (nombre, descripcion, categoria, ubicacion, factura, proveedor, tipo,
                                 area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (asset.nombre, asset.descripcion, asset.categoria, asset.ubicacion, asset.factura, asset.proveedor,
              asset.tipo, asset.area_responsable, asset.fecha_ingreso, asset.valor_adquisicion, asset.estado, creado_en))
        new_id = cur.lastrowid
    return get_asset(new_id, user)

@app.put("/activos/{asset_id}")
def update_asset(asset_id: int, up: AssetUpdate, user=Depends(require_role(["supervisor","technical"]))):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT id FROM activos WHERE id=?", (asset_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Activo no encontrado")
        fields, values = [], []
        for k, v in up.dict(exclude_unset=True).items():
            fields.append(f"{k}=?"); values.append(v)
        if fields:
            q = f"UPDATE activos SET {', '.join(fields)} WHERE id=?"
            values.append(asset_id)
            cur.execute(q, tuple(values))
    return {"message": "Activo actualizado"}

@app.delete("/activos/{asset_id}")
def delete_asset(asset_id: int, user=Depends(require_role(["management"]))):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM activos WHERE id=?", (asset_id,))
    return {"message": "Activo eliminado"}

# =========================
# EXPORT PARA POWER BI (PROTEGIDO POR JWT)
# =========================
@app.get("/export/activos.csv")
def export_csv(user=Depends(get_current_user)):
    rows = fetch_export_rows()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_COLUMNS)
    writer.writerows(rows)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="activos.csv"'})

@app.get("/export/activos.json")
def export_json(user=Depends(get_current_user)):
    rows = fetch_export_rows()
    data = [dict(zip(EXPORT_COLUMNS, r)) for r in rows]
    return {"data": data}

# =========================
# EXPORT PÚBLICO PARA POWER BI (API KEY por querystring)
# =========================
EXPORT_API_KEY = os.getenv("EXPORT_API_KEY", "dev-export-key")

def check_export_key(request: Request):
    key = request.query_params.get("key")
    if key != EXPORT_API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")

# Ping protegido: para probar rápidamente la key
@app.get("/export/public/ping")
def export_ping(request: Request):
    check_export_key(request)
    return {"ok": True, "msg": "key válida"}

@app.get("/export/public/activos.csv")
def export_csv_public(request: Request):
    check_export_key(request)
    try:
        rows = fetch_export_rows()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(EXPORT_COLUMNS)
        if rows:
            writer.writerows(rows)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="activos.csv"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR export_csv_public:", repr(e))
        raise HTTPException(status_code=500, detail="Error generando CSV")

@app.get("/export/public/activos.json")
def export_json_public(request: Request):
    check_export_key(request)
    try:
        rows = fetch_export_rows()
        data = [dict(zip(EXPORT_COLUMNS, r)) for r in rows]
        return {"data": data}
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR export_json_public:", repr(e))
        raise HTTPException(status_code=500, detail="Error generando JSON")

# =========================
# DEBUG (no aparecen en /docs)
# =========================
@app.get("/_debug/routes", include_in_schema=False)
def _debug_routes():
    try:
        return {"routes": [getattr(r, "path", str(r)) for r in app.routes]}
    except Exception as e:
        return {"ok": False, "err": type(e)._name_, "msg": str(e)}

@app.get("/_debug/export/check", include_in_schema=False)
def _debug_export_check():
    """
    NO requiere API key. Solo comprueba que fetch_export_rows() funciona.
    """
    try:
        rows = fetch_export_rows()
        return {
            "ok": True,
            "columns": EXPORT_COLUMNS,
            "row_count": len(rows),
            "sample": rows[:3],
        }
    except Exception as e:
        import traceback
        return {
            "ok": False,
            "error_type": type(e)._name_,
            "error_msg": str(e),
            "trace": traceback.format_exc().splitlines()[-8:],
        }

@app.post("/_debug/seed", include_in_schema=False)
def _debug_seed(request: Request):
    """
    Inserta 2 filas demo. Protegido con la misma API key pública del export.
    """
    check_export_key(request)  # requiere ?key=...
    from random import randint
    now = datetime.utcnow().isoformat()
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""INSERT INTO activos 
            (nombre, descripcion, categoria, ubicacion, factura, proveedor, tipo, 
             area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"Activo Demo {randint(100,999)}", "Equipo de prueba", "IT", "Almacén A",
             "F-00123", "Proveedor X", "Hardware", "Sistemas", "2025-08-01", 1200.50, "activo", now)
        )
        cur.execute("""INSERT INTO activos 
            (nombre, descripcion, categoria, ubicacion, factura, proveedor, tipo, 
             area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"Activo Demo {randint(100,999)}", "Equipo de prueba", "Infraestructura", "Planta 1",
             "F-00456", "Proveedor Y", "Maquinaria", "Mantenimiento", "2025-07-15", 5400.00, "activo", now)
        )
    return {"ok": True, "msg": "Semilla insertada"}

# =========================
# IMPORT CSV + WIPE + MIGRACIÓN (endpoints administrativos)
# =========================
def _normalize_date(value: str) -> str:
    """
    Normaliza fechas a YYYY-MM-DD cuando es posible.
    Acepta 'YYYY-MM-DD', 'YYYY/MM/DD', 'DD/MM/YYYY', 'MM/DD/YYYY'.
    Si no reconoce, devuelve tal cual.
    """
    if not value:
        return ""
    v = str(value).strip().replace("\\", "/")
    try:
        # YYYY-MM-DD
        if len(v) >= 10 and v[4] == "-" and v[7] == "-":
            return v[:10]
        # YYYY/MM/DD
        if len(v) >= 10 and v[4] == "/" and v[7] == "/":
            y, m, d = v[:10].split("/")
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        # DD/MM/YYYY
        if "/" in v and len(v) >= 10 and v.count("/") == 2:
            d, m, y = v[:10].split("/")
            if len(y) == 4:
                return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        # MM/DD/YYYY
        if "-" not in v and "/" in v and len(v) >= 10:
            m, d, y = v[:10].split("/")
            if len(y) == 4:
                return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    except Exception:
        pass
    return v

def _to_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.replace(" ", "")
    # 1.234,56 -> 1234.56
    if s.count(",") == 1 and s.count(".") > 1:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def _rows_from_csv(file_stream) -> Iterable[dict]:
    """
    Lee un CSV (utf-8/utf-8-sig) y mapea cabeceras comunes a nuestro esquema.
    Cabeceras objetivo: nombre, descripcion, categoria, ubicacion, factura, proveedor,
    tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en.
    """
    wrapper = TextIOWrapper(file_stream, encoding="utf-8-sig")
    reader = csv.DictReader(wrapper)
    header_map = {
        # iguales
        "id": "id",
        "nombre": "nombre",
        "descripcion": "descripcion",
        "categoria": "categoria",
        "ubicacion": "ubicacion",
        "factura": "factura",
        "proveedor": "proveedor",
        "tipo": "tipo",
        "area_responsable": "area_responsable",
        "fecha_ingreso": "fecha_ingreso",
        "valor_adquisicion": "valor_adquisicion",
        "estado": "estado",
        "creado_en": "creado_en",
        # variantes típicas
        "numero de activo fijo": "nombre",
        "nombre del activo": "nombre",
        "grupo": "categoria",
        "conjunto de activos fijos": "categoria",
        "departamento": "ubicacion",
        "custodio": "area_responsable",
        "libro": "tipo",
        "tipo de libro": "tipo",
        "fecha de adquisición": "fecha_ingreso",
        "precio de adquisición": "valor_adquisicion",
        "valor neto en libros": "valor_adquisicion",
    }
    for raw in reader:
        # fila completamente vacía -> saltar
        if not any((raw.get(k) or "").strip() for k in raw.keys()):
            continue
        out = {
            "nombre": None,
            "descripcion": (raw.get("descripcion") or "").strip(),
            "categoria": None,
            "ubicacion": None,
            "factura": (raw.get("factura") or "").strip(),
            "proveedor": (raw.get("proveedor") or "").strip(),
            "tipo": None,
            "area_responsable": None,
            "fecha_ingreso": "",
            "valor_adquisicion": None,
            "estado": (raw.get("estado") or "").strip() or "activo",
            "creado_en": (raw.get("creado_en") or "").strip() or datetime.utcnow().isoformat(),
        }
        for k, v in list(raw.items()):
            if v is None:
                continue
            key = (k or "").strip().lower()
            mapped = header_map.get(key)
            if mapped:
                if mapped == "valor_adquisicion":
                    out["valor_adquisicion"] = _to_float(v)
                elif mapped == "fecha_ingreso":
                    out["fecha_ingreso"] = _normalize_date(v)
                elif mapped == "nombre" and not out["nombre"]:
                    out["nombre"] = str(v).strip()
                elif mapped in out:
                    out[mapped] = str(v).strip()
        if not out["nombre"]:
            out["nombre"] = (raw.get("nombre") or raw.get("descripcion") or "Activo").strip()
        yield out

@app.post("/_debug/import/csv", include_in_schema=False)
async def _debug_import_csv(request: Request):
    """
    Sube un CSV (form-data → file) e inserta filas en 'activos'.
    Requiere ?key=EXPORT_API_KEY.
    """
    check_export_key(request)
    form = await request.form()
    if "file" not in form:
        raise HTTPException(status_code=400, detail="Sube el archivo CSV en form-data con campo 'file'")
    file = form["file"]
    inserted = 0
    with get_conn() as c:
        cur = c.cursor()
        for row in _rows_from_csv(file.file):
            cur.execute("""
                INSERT INTO activos
                (nombre, descripcion, categoria, ubicacion, factura, proveedor, tipo,
                 area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                row["nombre"], row["descripcion"], row["categoria"], row["ubicacion"],
                row["factura"], row["proveedor"], row["tipo"], row["area_responsable"],
                row["fecha_ingreso"], row["valor_adquisicion"], row["estado"], row["creado_en"]
            ))
            inserted += 1
    return {"ok": True, "inserted": inserted}

@app.post("/_debug/wipe", include_in_schema=False)
def _debug_wipe(request: Request):
    """Borra todos los registros de 'activos'. Requiere ?key=EXPORT_API_KEY."""
    check_export_key(request)
    with get_conn() as c:
        c.execute("DELETE FROM activos")
    return {"ok": True, "message": "Tabla 'activos' vaciada"}

@app.post("/_debug/migrate", include_in_schema=False)
def _debug_migrate(request: Request):
    """Ejecuta la migración de columnas faltantes en 'activos'. Requiere ?key=EXPORT_API_KEY."""
    check_export_key(request)
    ensure_activos_columns()
    return {"ok": True, "message": "Migración aplicada"}

# ===== BLOQUE B — IA HELPERS (features, reglas, regresión) =====
def _years_since(yyyy_mm_dd: str) -> float:
    if not yyyy_mm_dd:
        return 0.0
    try:
        y, m, d = (yyyy_mm_dd[:10].split("-") + ["01","01"])[:3]
        y = int(y); m = int(m or "1"); d = int(d or "1")
        dt = datetime(y, m, d)
        return max(0.0, (datetime.utcnow() - dt).days / 365.25)
    except Exception:
        return 0.0

def _safe_float(x):
    try:
        return float(x) if x is not None and str(x).strip() != "" else 0.0
    except Exception:
        return 0.0

def _norm_log1p(x: float) -> float:
    # normaliza log(1+x) a ~[0..1] suponiendo valores hasta 1e6
    return math.log1p(max(0.0, x)) / math.log1p(1_000_000)

def _feature_vector(asset: dict) -> dict:
    """
    Devuelve un dict de características ESTABLES (no cambiar nombres una vez publicados).
    """
    valor = _safe_float(asset.get("valor_adquisicion"))
    antig = _years_since(asset.get("fecha_ingreso") or "")
    estado = (asset.get("estado") or "").strip().lower()
    desc = (asset.get("descripcion") or "")
    nombre = (asset.get("nombre") or "")
    proveedor = (asset.get("proveedor") or "")
    factura = (asset.get("factura") or "")

    return {
        "f_valor_log": _norm_log1p(valor),
        "f_antiguedad_anios": min(25.0, antig) / 25.0,        # cap a 25 años
        "f_estado_no_activo": 1.0 if estado not in ("activo", "") else 0.0,
        "f_tiene_factura": 1.0 if factura else 0.0,
        "f_tiene_proveedor": 1.0 if proveedor else 0.0,
        "f_len_nombre": min(50, len(nombre)) / 50.0,
        "f_len_desc": min(200, len(desc)) / 200.0,
    }

def _vector_to_list(feats: dict, feature_names: list) -> list:
    return [feats.get(k, 0.0) for k in feature_names]

def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0/(1.0+ez)
    else:
        ez = math.exp(z)
        return ez/(1.0+ez)

def _predict_proba(weights: list, x: list) -> float:
    # weights y x incluyen bias en x[0]=1.0
    z = 0.0
    for w, xi in zip(weights, x):
        z += w * xi
    return _sigmoid(z)

def _rule_score(asset: dict) -> float:
    """
    Heurística por si no hay modelo entrenado.
    """
    score = 0.0
    antig = _years_since(asset.get("fecha_ingreso") or "")
    estado = (asset.get("estado") or "").strip().lower()
    valor = _safe_float(asset.get("valor_adquisicion"))
    if antig >= 8: score += 0.45
    if estado in ("baja", "obsoleto", "averiado"): score += 0.45
    if valor <= 500: score += 0.15
    return min(1.0, score)

def _get_ai_threshold_default() -> float:
    return float(_get_setting("AI_THRESHOLD_CRITICAL", 0.5))

def _load_model():
    weights = _get_setting("AI_LOGREG_WEIGHTS", None)  # list de floats, incluye bias
    feats = _get_setting("AI_FEATURE_NAMES", None)     # orden de features (sin bias)
    thr = float(_get_setting("AI_THRESHOLD_CRITICAL", 0.5))
    return weights, feats, thr

# ===== BLOQUE C — IA ENDPOINTS (labels, train, predict, ver modelo) =====
class TrainRequest(BaseModel):
    epochs: Optional[int] = 300
    lr: Optional[float] = 0.05
    l2: Optional[float] = 0.001
    threshold: Optional[float] = None  # si no, mantiene o 0.5

class LabelBody(BaseModel):
    label: int  # 0 = ok, 1 = critico

@app.post("/ia/labels/{asset_id}")
def ia_add_label(asset_id: int, body: LabelBody, user=Depends(require_role(["supervisor","management"]))):
    if body.label not in (0,1):
        raise HTTPException(status_code=400, detail="label debe ser 0 o 1")
    # validamos que exista el activo
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT id FROM activos WHERE id=?", (asset_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Activo no encontrado")
        cur.execute("""INSERT INTO ml_labels (asset_id,label,timestamp) 
                       VALUES(?,?,?)""", (asset_id, body.label, datetime.utcnow().isoformat()))
    return {"ok": True, "asset_id": asset_id, "label": body.label}

@app.post("/ia/train")
def ia_train(req: TrainRequest, user=Depends(require_role(["supervisor","management"]))):
    # 1) Cargar dataset etiquetado
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""
            SELECT a.id, a.nombre, a.descripcion, a.categoria, a.ubicacion, a.factura, a.proveedor,
                   a.tipo, a.area_responsable, a.fecha_ingreso, a.valor_adquisicion, a.estado, a.creado_en,
                   l.label
            FROM ml_labels l
            JOIN activos a ON a.id = l.asset_id
        """)
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(status_code=400, detail="No hay etiquetas en ml_labels. Crea algunas con /ia/labels/{id}")

    cols = ["id","nombre","descripcion","categoria","ubicacion","factura","proveedor",
            "tipo","area_responsable","fecha_ingreso","valor_adquisicion","estado","creado_en","label"]

    dataset = [dict(zip(cols, r)) for r in rows]

    # 2) Construir features y y
    feature_names = ["f_valor_log","f_antiguedad_anios","f_estado_no_activo",
                     "f_tiene_factura","f_tiene_proveedor","f_len_nombre","f_len_desc"]

    X = []
    y = []
    for d in dataset:
        feats = _feature_vector(d)
        X.append([1.0] + _vector_to_list(feats, feature_names))  # bias
        y.append(int(d["label"]))

    n, m = len(X), len(X[0])  # n muestras, m features (incl. bias)
    # 3) Entrenar regresión logística con GD
    w = [0.0]*m
    lr = float(req.lr or 0.05)
    l2 = float(req.l2 or 0.001)
    epochs = int(req.epochs or 300)

    for _ in range(epochs):
        grad = [0.0]*m
        for xi, yi in zip(X, y):
            p = _predict_proba(w, xi)
            err = p - yi
            for j in range(m):
                grad[j] += err * xi[j]
        # regularización L2 (no penalizar bias j=0)
        for j in range(1, m):
            grad[j] += l2 * w[j]
        for j in range(m):
            w[j] -= (lr / n) * grad[j]

    # 4) Métricas rápidas
    tp=fp=tn=fn=0
    thr = float(req.threshold) if req.threshold is not None else _get_ai_threshold_default()
    for xi, yi in zip(X, y):
        p = _predict_proba(w, xi)
        yhat = 1 if p >= thr else 0
        if yi==1 and yhat==1: tp+=1
        elif yi==0 and yhat==1: fp+=1
        elif yi==0 and yhat==0: tn+=1
        elif yi==1 and yhat==0: fn+=1
    acc = (tp+tn)/max(1,(tp+tn+fp+fn))
    prec = tp/max(1,(tp+fp))
    rec = tp/max(1,(tp+fn))
    f1 = 2*prec*rec/max(1e-9,(prec+rec))

    # 5) Guardar modelo
    _set_setting("AI_LOGREG_WEIGHTS", w)
    _set_setting("AI_FEATURE_NAMES", feature_names)
    _set_setting("AI_THRESHOLD_CRITICAL", thr)

    _save_model_file(w, feature_names, thr)  # <-- NUEVO: copia best-effort a archivo

    return {
        "ok": True,
        "samples": n,
        "features": feature_names,
        "weights_len": len(w),
        "threshold_usado": thr,
        "metrics": {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
    }

@app.get("/ia/model")
def ia_model(user=Depends(require_role(["supervisor","management"]))):
    w, feats, thr = _load_model()
    return {
        "has_model": bool(w and feats),
        "feature_names": feats or [],
        "weights": w or [],
        "threshold": thr
    }

# ===== NUEVO: endpoints opcionales para forzar guardar/cargar el archivo del modelo =====
@app.post("/ia/persist/save")
def ia_persist_save(user=Depends(require_role(["supervisor","management"]))):
    w, feats, thr = _load_model()
    if not (w and feats):
        raise HTTPException(status_code=400, detail="No hay modelo en DB para guardar")
    ok = _save_model_file(w, feats, thr)
    return {"ok": bool(ok), "path": MODEL_PATH}

@app.post("/ia/persist/load")
def ia_persist_load(user=Depends(require_role(["supervisor","management"]))):
    w_f, f_f, t_f = _load_model_file()
    if not (w_f and f_f):
        raise HTTPException(status_code=400, detail="No se encontró modelo.pkl válido")
    _set_setting("AI_LOGREG_WEIGHTS", w_f)
    _set_setting("AI_FEATURE_NAMES", f_f)
    _set_setting("AI_THRESHOLD_CRITICAL", float(t_f if t_f is not None else 0.5))
    return {"ok": True, "imported_from_file": True}

@app.get("/ia/predict/{asset_id}")
def ia_predict(asset_id: int, user=Depends(get_current_user)):
    # cargar activo
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT id, nombre, descripcion, categoria, ubicacion, factura, proveedor,
                              tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en
                       FROM activos WHERE id=?""", (asset_id,))
        r = cur.fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Activo no encontrado")
    cols = ["id","nombre","descripcion","categoria","ubicacion","factura","proveedor",
            "tipo","area_responsable","fecha_ingreso","valor_adquisicion","estado","creado_en"]
    a = dict(zip(cols, r))

    w, feats, thr = _load_model()
    if w and feats:
        x = [1.0] + _vector_to_list(_feature_vector(a), feats)
        proba = _predict_proba(w, x)
        pred = 1 if proba >= thr else 0
        src = "model"
    else:
        proba = _rule_score(a)
        pred = 1 if proba >= _get_ai_threshold_default() else 0
        src = "rules"

    return {
        "asset_id": asset_id,
        "proba_critico": proba,
        "threshold": thr if src=="model" else _get_ai_threshold_default(),
        "pred_label": pred,
        "source": src
    }

# =========================
# ENDPOINT OPTIMIZADO PARA POWER BI (KPIs + DETALLE) — BLOQUE D con IA opcional
# =========================
@app.get("/reportes/activos")
def reportes_activos(
    include_ia: int = 0,
    include_ia_model: int = 0,
    user=Depends(get_current_user)
):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT id, nombre, categoria, ubicacion, tipo, area_responsable,
                              fecha_ingreso, valor_adquisicion, estado, creado_en, descripcion, factura, proveedor
                       FROM activos""")
        rows = cur.fetchall()
    cols = ["id","nombre","categoria","ubicacion","tipo","area_responsable",
            "fecha_ingreso","valor_adquisicion","estado","creado_en","descripcion","factura","proveedor"]
    data = [dict(zip(cols, r)) for r in rows]

    # IA opcional
    attach_ia = (include_ia or include_ia_model)
    w, feats, thr = _load_model()
    use_model = bool(w and feats) if include_ia_model else False

    if attach_ia:
        for d in data:
            if use_model:
                x = [1.0] + _vector_to_list(_feature_vector(d), feats)
                p = _predict_proba(w, x)
                t = thr
                src = "model"
            else:
                p = _rule_score(d)
                t = _get_ai_threshold_default()
                src = "rules"
            d["ia_score"] = p
            d["ia_label"] = 1 if p >= t else 0
            d["ia_source"] = src

    total = len(data)
    activos = sum(1 for d in data if (d["estado"] or "").lower() == "activo")
    bajas = sum(1 for d in data if (d["estado"] or "").lower() == "baja")

    por_estado, por_categoria, por_mes = {}, {}, {}
    for d in data:
        key_e = (d["estado"] or "desconocido").lower()
        por_estado[key_e] = por_estado.get(key_e, 0) + 1
        key_c = (d["categoria"] or "sin_categoria")
        por_categoria[key_c] = por_categoria.get(key_c, 0) + 1
        fi = (d["fecha_ingreso"] or "")
        ym = fi[:7] if len(fi) >= 7 else "sin_fecha"
        por_mes[ym] = por_mes.get(ym, 0) + 1

    # Limpiar extras no usados en payload base
    for d in data:
        d.pop("descripcion", None)
        d.pop("factura", None)
        d.pop("proveedor", None)

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
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT id, nombre, descripcion, categoria, ubicacion, factura, proveedor,
                              tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en
                       FROM activos WHERE id=?""", (asset_id,))
        r = cur.fetchone()
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
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("UPDATE activos SET estado=? WHERE id=?", ("baja", asset_id))
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
        headers={"Content-Disposition": f'attachment; filename="acta_baja_{asset_id}.pdf"'})

@app.get("/reportes/instantaneo.pdf")
def reporte_ejecutivo_pdf(user=Depends(get_current_user)):
    rep = reportes_activos(user=user)  # reutiliza el endpoint optimizado (sin IA por defecto)
    k = rep["kpis"]; agg = rep["agregados"]

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
