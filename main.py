from fastapi import FastAPI, HTTPException, Depends, Request 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta, date
import os, sqlite3, io
import csv, json, math
from io import TextIOWrapper
from typing import Iterable
import joblib  # <-- para guardar/cargar el modelo a archivo
import shutil  # <-- para migrar archivos a /data
from calendar import monthrange



# Render sin GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# APP BASE
# =========================
APP_VERSION = "1.3.5"
app = FastAPI(title="API Gestión de Activos", version=APP_VERSION)

# -------- CORS (soporta previews de Lovable) --------
# ORIGINS: lista explícita separada por comas (localhost, tu backend, tu front oficial si lo tienes)
origins_env = os.getenv("ORIGINS", "*")
ORIGINS = [""] if origins_env.strip() == "" else [o.strip() for o in origins_env.split(",") if o.strip()]

# ORIGINS_REGEX: acepta cualquier subdominio de Lovable (*.lovable.page)
# puedes sobreescribirlo por env si quieres restringir más
ORIGINS_REGEX = os.getenv("ORIGINS_REGEX", r"https://.*\.lovable\.page$")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=ORIGINS_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------------


# =========================
# RUTAS PERSISTENTES (DB y MODELO) + MIGRACIÓN A /data
# =========================
DEFAULT_DB_FILE = "activos.db"
DEFAULT_MODEL_FILE = "modelo.pkl"

def _in_data_dir() -> bool:
    return os.path.isdir("/data")

DB_NAME = os.getenv("DB_NAME", f"/data/{DEFAULT_DB_FILE}" if _in_data_dir() else DEFAULT_DB_FILE)
MODEL_PATH = os.getenv("MODEL_PATH", f"/data/{DEFAULT_MODEL_FILE}" if _in_data_dir() else DEFAULT_MODEL_FILE)

def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent and parent != "":
        os.makedirs(parent, exist_ok=True)

def _maybe_migrate_to_data(src_name: str, dest_path: str):
    try:
        if dest_path.startswith("/data/"):
            _ensure_parent(dest_path)
            if (not os.path.exists(dest_path)) and os.path.exists(src_name):
                shutil.copy2(src_name, dest_path)
                print(f"MIGRATE: copiado {src_name} -> {dest_path}")
    except Exception as e:
        print("WARN migrate:", src_name, "->", dest_path, repr(e))

def _preflight_persistence():
    _maybe_migrate_to_data(os.path.basename(DB_NAME) or DEFAULT_DB_FILE, DB_NAME)
    _maybe_migrate_to_data(os.path.basename(MODEL_PATH) or DEFAULT_MODEL_FILE, MODEL_PATH)

_preflight_persistence()

# =========================
# DB
# =========================
def get_conn():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    with get_conn() as c:
        cur = c.cursor()
        # Tabla de activos (agregamos vida útil y método de depreciación)
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
            creado_en TEXT,
            vida_util_anios REAL DEFAULT 5.0,
            metodo_depreciacion TEXT DEFAULT 'linea_recta'
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
            label INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES activos(id)
        )
        """)

def ensure_activos_columns():
    """
    Migra la tabla 'activos' si faltan columnas de versiones anteriores.
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
        # Nuevos para depreciación:
        "vida_util_anios": "REAL DEFAULT 5.0",
        "metodo_depreciacion": "TEXT DEFAULT 'linea_recta'",
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

ensure_settings_table()

# ===== Persistencia a archivo del modelo =====
def _save_model_file(weights: list, features: list, threshold: float):
    payload = {
        "weights": weights,
        "features": features,
        "threshold": float(threshold),
        "version": APP_VERSION,
        "saved_at": datetime.utcnow().isoformat(),
        "algo": "logreg-gd",
    }
    try:
        _ensure_parent(MODEL_PATH)
        joblib.dump(payload, MODEL_PATH)
        return True
    except Exception as e:
        print("WARN: no se pudo guardar modelo.pkl:", repr(e))
        return False

def _load_model_file():
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None, None
        payload = joblib.load(MODEL_PATH)
        return payload.get("weights"), payload.get("features"), float(payload.get("threshold", 0.5))
    except Exception as e:
        print("WARN: no se pudo cargar modelo.pkl:", repr(e))
        return None, None, None

def _sync_model_persistence():
    w_db = _get_setting("AI_LOGREG_WEIGHTS", None)
    f_db = _get_setting("AI_FEATURE_NAMES", None)
    t_db = _get_setting("AI_THRESHOLD_CRITICAL", None)

    w_f, f_f, t_f = _load_model_file()

    if (w_f and f_f) and not (w_db and f_db):
        _set_setting("AI_LOGREG_WEIGHTS", w_f)
        _set_setting("AI_FEATURE_NAMES", f_f)
        _set_setting("AI_THRESHOLD_CRITICAL", float(t_f if t_f is not None else 0.5))
        print("SYNC: importado modelo desde archivo -> DB")
        return

    if (w_db and f_db) and not (w_f and f_f):
        _save_model_file(w_db, f_db, float(t_db if t_db is not None else 0.5))
        print("SYNC: exportado modelo desde DB -> archivo")
        return

    print("SYNC: modelo presente en DB (y archivo opcional)")

@app.on_event("startup")
def _startup_model_sync():
    try:
        _sync_model_persistence()
    except Exception as e:
        print("WARN startup sync:", repr(e))

# =========================
# MODELOS (Pydantic)
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
    # Nuevos campos (opcionales; por defecto no rompen nada):
    vida_util_anios: Optional[float] = 5.0
    metodo_depreciacion: Optional[str] = "linea_recta"

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
    # Nuevos:
    vida_util_anios: Optional[float]
    metodo_depreciacion: Optional[str]

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
    # Nuevos:
    vida_util_anios: Optional[float]
    metodo_depreciacion: Optional[str]

class UserCreate(BaseModel):
    email: str
    password: str
    role: Optional[str] = None

class BajaRequest(BaseModel):
    motivo: str

# =========================
# AUTH + ROLES (JWT)
# =========================
from passlib.hash import pbkdf2_sha256 as hasher
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "devsecret-change-me")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

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

# Incluir nuevas columnas en los SELECT para no romper el response_model
_COMMON_SELECT = """SELECT id, nombre, descripcion, categoria, ubicacion, factura, proveedor,
                           tipo, area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en,
                           vida_util_anios, metodo_depreciacion
                    FROM activos"""

@app.get("/activos", response_model=List[AssetOut])
def get_assets(user=Depends(get_current_user)):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(_COMMON_SELECT + " ORDER BY id DESC")
        rows = cur.fetchall()
    return [AssetOut(
        id=r[0], nombre=r[1], descripcion=r[2], categoria=r[3], ubicacion=r[4],
        factura=r[5], proveedor=r[6], tipo=r[7], area_responsable=r[8],
        fecha_ingreso=r[9], valor_adquisicion=r[10], estado=r[11], creado_en=r[12],
        vida_util_anios=r[13], metodo_depreciacion=r[14]
    ) for r in rows]

@app.get("/activos/{asset_id}", response_model=AssetOut)
def get_asset(asset_id: int, user=Depends(get_current_user)):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(_COMMON_SELECT + " WHERE id=?", (asset_id,))
        r = cur.fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Activo no encontrado")
    return AssetOut(
        id=r[0], nombre=r[1], descripcion=r[2], categoria=r[3], ubicacion=r[4],
        factura=r[5], proveedor=r[6], tipo=r[7], area_responsable=r[8],
        fecha_ingreso=r[9], valor_adquisicion=r[10], estado=r[11], creado_en=r[12],
        vida_util_anios=r[13], metodo_depreciacion=r[14]
    )

from fastapi import Body, Depends, HTTPException
from datetime import datetime
from typing import Any, Dict, Set
from sqlite3 import Connection

# --------- helpers muy simples ---------
def _table_columns(c: Connection, table: str) -> Set[str]:
    cur = c.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}

def _clean_str(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None

def _parse_date(v):
    if not v:
        return None
    if isinstance(v, (datetime,)):
        return v.date().isoformat()
    s = str(v).strip()
    if not s:
        return None
    from datetime import datetime as dt
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return dt.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    # último intento: ISO con hora
    try:
        return dt.fromisoformat(s).date().isoformat()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"fecha_adquisicion inválida: '{v}'")

def _parse_number(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    # Soporta "1.234,56" y "1,234.56"
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(s.replace(".", "").replace(",", "."))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"valor numérico inválido: '{v}'")

def _build_row_from_front(payload: Dict[str, Any], cols_in_db: Set[str]) -> Dict[str, Any]:
    """
    Mapea las claves del front a columnas reales y normaliza valores.
    Front manda:
      nombre, codigo, categoria, descripcion, ubicacion, estado,
      fecha_adquisicion, precio_inicial, vida_util_anos, valor_residual, responsable_email
    BD (ejemplo):
      nombre, descripcion, categoria, ubicacion, estado,
      fecha_ingreso, valor_adquisicion, vida_util_anios,
      (opcionales) codigo, valor_residual, responsable_email, creado_en, ...
    """
    # 1) sacar valores limpios
    nombre = _clean_str(payload.get("nombre"))
    if not nombre:
        raise HTTPException(400, detail="El campo 'nombre' es obligatorio.")

    row: Dict[str, Any] = {
        "nombre": nombre,
        "descripcion": _clean_str(payload.get("descripcion")),
        "categoria": _clean_str(payload.get("categoria")),
        "ubicacion": _clean_str(payload.get("ubicacion")),
        "estado": _clean_str(payload.get("estado")),
        "fecha_ingreso": _parse_date(payload.get("fecha_adquisicion")),
        "valor_adquisicion": _parse_number(payload.get("precio_inicial")),
        "vida_util_anios": _parse_number(payload.get("vida_util_anos")),
        # opcionales que quizás existan en la tabla
        "codigo": _clean_str(payload.get("codigo")),
        "valor_residual": _parse_number(payload.get("valor_residual")),
        "responsable_email": _clean_str(payload.get("responsable_email")),
        # timestamps
        "creado_en": datetime.utcnow().isoformat(),
    }

    # 2) quita las columnas que NO existen realmente en la tabla
    row = {k: v for k, v in row.items() if k in cols_in_db}

    # 3) algunos campos que tu tabla podría exigir (pero el front no manda)
    # si existen en la tabla y no tienes valor, los dejamos en None:
    for maybe in ("tipo", "area_responsable", "factura", "proveedor", "metodo_depreciacion"):
        if maybe in cols_in_db and maybe not in row:
            row[maybe] = None

    return row

# --------- Ruta súper simple: dict crudo + normalizador ---------
@app.post("/activos", response_model=AssetOut, status_code=201)
def create_asset(
    asset: Dict[str, Any] = Body(...),
    user = Depends(require_role(["storekeeper", "supervisor", "management"])),
):
    with get_conn() as c:
        cols = _table_columns(c, "activos")
        row = _build_row_from_front(asset, cols)

        columns = list(row.keys())
        if not columns:
            raise HTTPException(500, detail="No hay columnas válidas para insertar.")

        placeholders = ", ".join(["?"] * len(columns))
        columns_sql = ", ".join(columns)
        values = [row[k] for k in columns]

        cur = c.cursor()
        cur.execute(f"INSERT INTO activos ({columns_sql}) VALUES ({placeholders})", values)
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
        return {"ok": False, "err": type(e).__name__, "msg": str(e)}

@app.get("/_debug/export/check", include_in_schema=False)
def _debug_export_check():
    try:
        rows = fetch_export_rows()
        return {"ok": True, "columns": EXPORT_COLUMNS, "row_count": len(rows), "sample": rows[:3]}
    except Exception as e:
        import traceback
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "trace": traceback.format_exc().splitlines()[-8:],
        }

@app.post("/_debug/seed", include_in_schema=False)
def _debug_seed(request: Request):
    check_export_key(request)
    from random import randint
    now = datetime.utcnow().isoformat()
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""INSERT INTO activos 
            (nombre, descripcion, categoria, ubicacion, factura, proveedor, tipo, 
             area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en,
             vida_util_anios, metodo_depreciacion)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"Activo Demo {randint(100,999)}", "Equipo de prueba", "IT", "Almacén A",
             "F-00123", "Proveedor X", "Hardware", "Sistemas", "2019-08-01", 1200.50, "activo", now,
             5.0, "linea_recta")
        )
        cur.execute("""INSERT INTO activos 
            (nombre, descripcion, categoria, ubicacion, factura, proveedor, tipo, 
             area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en,
             vida_util_anios, metodo_depreciacion)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"Activo Demo {randint(100,999)}", "Equipo de prueba", "Infraestructura", "Planta 1",
             "F-00456", "Proveedor Y", "Maquinaria", "Mantenimiento", "2022-07-15", 5400.00, "activo", now,
             10.0, "linea_recta")
        )
    return {"ok": True, "msg": "Semilla insertada"}

# =========================
# IMPORT CSV + WIPE + MIGRACIÓN (endpoints administrativos)
# =========================
def _normalize_date(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip().replace("\\", "/")
    try:
        if len(v) >= 10 and v[4] == "-" and v[7] == "-":
            return v[:10]
        if len(v) >= 10 and v[4] == "/" and v[7] == "/":
            y, m, d = v[:10].split("/")
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        if "/" in v and len(v) >= 10 and v.count("/") == 2:
            d, m, y = v[:10].split("/")
            if len(y) == 4:
                return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
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
    wrapper = TextIOWrapper(file_stream, encoding="utf-8-sig")
    reader = csv.DictReader(wrapper)
    header_map = {
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
        # Nuevos posibles encabezados:
        "vida_util_anios": "vida_util_anios",
        "vida útil (años)": "vida_util_anios",
        "metodo_depreciacion": "metodo_depreciacion",
        "método_depreciación": "metodo_depreciacion",
        # variantes
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
            "vida_util_anios": _to_float(raw.get("vida_util_anios")) if raw.get("vida_util_anios") else 5.0,
            "metodo_depreciacion": (raw.get("metodo_depreciacion") or "linea_recta").strip() or "linea_recta",
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
                elif mapped == "vida_util_anios":
                    out["vida_util_anios"] = _to_float(v) or 5.0
                elif mapped == "metodo_depreciacion":
                    out["metodo_depreciacion"] = str(v).strip() or "linea_recta"
                elif mapped == "nombre" and not out["nombre"]:
                    out["nombre"] = str(v).strip()
                elif mapped in out:
                    out[mapped] = str(v).strip()
        if not out["nombre"]:
            out["nombre"] = (raw.get("nombre") or raw.get("descripcion") or "Activo").strip()
        yield out

@app.post("/_debug/import/csv", include_in_schema=False)
async def _debug_import_csv(request: Request):
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
                 area_responsable, fecha_ingreso, valor_adquisicion, estado, creado_en,
                 vida_util_anios, metodo_depreciacion)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                row["nombre"], row["descripcion"], row["categoria"], row["ubicacion"],
                row["factura"], row["proveedor"], row["tipo"], row["area_responsable"],
                row["fecha_ingreso"], row["valor_adquisicion"], row["estado"], row["creado_en"],
                row["vida_util_anios"], row["metodo_depreciacion"]
            ))
            inserted += 1
    return {"ok": True, "inserted": inserted}

@app.post("/_debug/wipe", include_in_schema=False)
def _debug_wipe(request: Request):
    check_export_key(request)
    with get_conn() as c:
        c.execute("DELETE FROM activos")
    return {"ok": True, "message": "Tabla 'activos' vaciada"}

@app.post("/_debug/migrate", include_in_schema=False)
def _debug_migrate(request: Request):
    check_export_key(request)
    ensure_activos_columns()
    return {"ok": True, "message": "Migración aplicada"}

# =========================
# MANTENIMIENTO — Helpers
# =========================

def _dt_from_txt(s: str):
    try:
        return datetime.strptime((s or "")[:10], "%Y-%m-%d")
    except Exception:
        return None

def _add_months(dt: date, months: int) -> date:
    y = dt.year + (dt.month - 1 + months)//12
    m = (dt.month - 1 + months)%12 + 1
    d = min(dt.day, monthrange(y, m)[1])
    return date(y, m, d)

def _months_between(d1: date, d2: date) -> int:
    # meses (aprox.) entre fechas
    return max(0, (d2.year - d1.year)*12 + (d2.month - d1.month))

def _base_interval_months(cat: str, tipo: str) -> int:
    c = (cat or "").strip().lower()
    t = (tipo or "").strip().lower()
    if "maquin" in c or "vehic" in c or "maquin" in t or "vehic" in t:
        return 6       # maquinaria/vehículos
    if "infra" in c:
        return 12      # infraestructura
    if "hard" in t or "equipo" in t:
        return 12      # hardware/equipo
    return 12          # por defecto

def _status_from_due(next_due: date, today: date, soon_days: int):
    days = (next_due - today).days
    if days <= 0:
        return ("red", 0, days)       # vencido
    if days <= max(0, int(soon_days)):
        return ("yellow", 1, days)    # pronto
    return ("green", 2, days)         # ok

# =========================
# MANTENIMIENTO — Próximo por activo
# =========================
@app.get("/mantenimiento/proximo/{asset_id}")
def mantenimiento_proximo(asset_id: int, user=Depends(get_current_user)):
    # 1) Activo
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(_COMMON_SELECT + " WHERE id=?", (asset_id,))
        r = cur.fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Activo no encontrado")
    cols = ["id","nombre","descripcion","categoria","ubicacion","factura","proveedor",
            "tipo","area_responsable","fecha_ingreso","valor_adquisicion","estado","creado_en",
            "vida_util_anios","metodo_depreciacion"]
    a = dict(zip(cols, r))

    # 2) Riesgo (modelo si existe; si no, reglas)
    w, feats, thr = _load_model()
    if w and feats:
        x = [1.0] + _vector_to_list(_feature_vector(a), feats)
        proba = _predict_proba(w, x)
        src = "model"
    else:
        proba = _rule_score(a)
        src = "rules"

    # 3) Antigüedad/vida
    edad = _years_since(a.get("fecha_ingreso") or "")
    vida = float(a.get("vida_util_anios") or 0.0)
    age_frac = min(1.0, (edad / vida)) if vida > 0 else 0.0

    # 4) Intervalo ajustado
    base_meses = _base_interval_months(a.get("categoria"), a.get("tipo"))
    factor = 1.0 - 0.6*float(proba) - 0.3*age_frac
    if not math.isfinite(factor):
        factor = 1.0
    factor = max(0.3, min(1.2, factor))
    interval_meses = max(1, int(round(base_meses * factor)))

    # 5) Referencia y cálculo
    ref_dt = _dt_from_txt(a.get("fecha_ingreso")) or _dt_from_txt(a.get("creado_en")) or datetime.utcnow()
    ref = ref_dt.date()
    today = date.today()
    if ref >= today:
        next_due = _add_months(ref, interval_meses)
    else:
        n = _months_between(ref, today)
        k = (n // interval_meses) + 1
        next_due = _add_months(ref, k * interval_meses)

    return {
        "asset_id": a["id"],
        "categoria": a.get("categoria"),
        "tipo": a.get("tipo"),
        "edad_anios": round(edad, 2),
        "vida_util_anios": vida,
        "ia_proba_critico": round(float(proba), 4),
        "intervalo_base_meses": base_meses,
        "intervalo_ajustado_meses": interval_meses,
        "fecha_referencia": ref.isoformat(),
        "proxima_fecha_mantenimiento": next_due.isoformat(),
        "source": src
    }

# =========================
# MANTENIMIENTO — Agenda (para Dashboard)
# =========================
@app.get("/mantenimiento/agenda")
def mantenimiento_agenda(
    soon_days: int = 30,
    limit_overdue: int = 10,
    limit_soon: int = 10,
    limit_ok: int = 10,
    use_model: int = 1,
    user=Depends(get_current_user)
):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT id, nombre, categoria, ubicacion, tipo, area_responsable,
                              fecha_ingreso, valor_adquisicion, estado, creado_en,
                              vida_util_anios, metodo_depreciacion
                       FROM activos""")
        rows = cur.fetchall()

    cols = ["id","nombre","categoria","ubicacion","tipo","area_responsable",
            "fecha_ingreso","valor_adquisicion","estado","creado_en",
            "vida_util_anios","metodo_depreciacion"]
    items = [dict(zip(cols, r)) for r in rows]

    w, feats, thr = _load_model()
    use_model_bool = bool(w and feats) if use_model else False

    today = date.today()
    overdue, soon, ok = [], [], []

    for d in items:
        # proba
        if use_model_bool:
            x = [1.0] + _vector_to_list(_feature_vector(d), feats)
            p = float(_predict_proba(w, x))
        else:
            p = float(_rule_score(d))

        edad = _years_since(d.get("fecha_ingreso") or "")
        vida = float(d.get("vida_util_anios") or 0.0)
        age_frac = min(1.0, (edad / vida)) if vida > 0 else 0.0
        base_meses = _base_interval_months(d.get("categoria"), d.get("tipo"))

        factor = max(0.3, min(1.2, 1.0 - 0.6*p - 0.3*age_frac))
        interval_meses = max(1, int(round(base_meses * factor)))

        ref_dt = _dt_from_txt(d.get("fecha_ingreso")) or _dt_from_txt(d.get("creado_en")) or datetime.utcnow()
        ref = ref_dt.date()

        if ref >= today:
            next_due = _add_months(ref, interval_meses)
        else:
            n = _months_between(ref, today)
            k = (n // interval_meses) + 1
            next_due = _add_months(ref, k * interval_meses)

        status, severity, days_to_due = _status_from_due(next_due, today, soon_days)

        row = {
            "id": d["id"],
            "nombre": d["nombre"],
            "categoria": d["categoria"],
            "tipo": d["tipo"],
            "area_responsable": d["area_responsable"],
            "proxima_fecha_mantenimiento": next_due.isoformat(),
            "days_to_due": days_to_due,
            "maint_status": status,          # red|yellow|green
            "maint_severity": severity,      # 0|1|2
            "intervalo_mantenimiento_meses": interval_meses,
            "ia_score": round(p, 4),
        }

        if status == "red":
            overdue.append(row)
        elif status == "yellow":
            soon.append(row)
        else:
            ok.append(row)

    overdue.sort(key=lambda r: r["days_to_due"])   # más negativo primero
    soon.sort(key=lambda r: r["days_to_due"])      # más pronto primero
    ok.sort(key=lambda r: r["days_to_due"])

    return {
        "kpis": {
            "total_activos": len(items),
            "vencidos": len(overdue),
            "pronto": len(soon),
            "ok": len(ok),
            "ventana_pronto_dias": soon_days
        },
        "listas": {
            "overdue": overdue[:max(0, int(limit_overdue))],
            "soon":    soon[:max(0, int(limit_soon))],
            "ok":      ok[:max(0, int(limit_ok))]
        }
    }

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
    return math.log1p(max(0.0, x)) / math.log1p(1_000_000)

def _feature_vector(asset: dict) -> dict:
    valor = _safe_float(asset.get("valor_adquisicion"))
    antig = _years_since(asset.get("fecha_ingreso") or "")
    estado = (asset.get("estado") or "").strip().lower()
    desc = (asset.get("descripcion") or "")
    nombre = (asset.get("nombre") or "")
    proveedor = (asset.get("proveedor") or "")
    factura = (asset.get("factura") or "")

    return {
        "f_valor_log": _norm_log1p(valor),
        "f_antiguedad_anios": min(25.0, antig) / 25.0,
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
    z = 0.0
    for w, xi in zip(weights, x):
        z += w * xi
    return _sigmoid(z)

def _rule_score(asset: dict) -> float:
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
    weights = _get_setting("AI_LOGREG_WEIGHTS", None)
    feats = _get_setting("AI_FEATURE_NAMES", None)
    thr = float(_get_setting("AI_THRESHOLD_CRITICAL", 0.5))
    return weights, feats, thr

# ===== BLOQUE C — IA ENDPOINTS
class TrainRequest(BaseModel):
    epochs: Optional[int] = 300
    lr: Optional[float] = 0.05
    l2: Optional[float] = 0.001
    threshold: Optional[float] = None

class LabelBody(BaseModel):
    label: int

@app.post("/ia/labels/{asset_id}")
def ia_add_label(asset_id: int, body: LabelBody, user=Depends(require_role(["supervisor","management"]))):
    if body.label not in (0,1):
        raise HTTPException(status_code=400, detail="label debe ser 0 o 1")
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

    feature_names = ["f_valor_log","f_antiguedad_anios","f_estado_no_activo",
                     "f_tiene_factura","f_tiene_proveedor","f_len_nombre","f_len_desc"]

    X = []
    y = []
    for d in dataset:
        feats = _feature_vector(d)
        X.append([1.0] + _vector_to_list(feats, feature_names))
        y.append(int(d["label"]))

    n, m = len(X), len(X[0])
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
        for j in range(1, m):
            grad[j] += l2 * w[j]
        for j in range(m):
            w[j] -= (lr / n) * grad[j]

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

    _set_setting("AI_LOGREG_WEIGHTS", w)
    _set_setting("AI_FEATURE_NAMES", feature_names)
    _set_setting("AI_THRESHOLD_CRITICAL", thr)
    _save_model_file(w, feature_names, thr)

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
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(_COMMON_SELECT + " WHERE id=?", (asset_id,))
        r = cur.fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Activo no encontrado")
    cols = ["id","nombre","descripcion","categoria","ubicacion","factura","proveedor","tipo","area_responsable",
            "fecha_ingreso","valor_adquisicion","estado","creado_en","vida_util_anios","metodo_depreciacion"]
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
# ENDPOINT KPIs + DETALLE (con IA y Mantenimiento opcional)
# =========================
@app.get("/reportes/activos")
def reportes_activos(
    include_ia: int = 0,
    include_ia_model: int = 0,
    include_maintenance: int = 0,    # NUEVO
    soon_days: int = 30,              # NUEVO
    user=Depends(get_current_user)
):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT id, nombre, categoria, ubicacion, tipo, area_responsable,
                              fecha_ingreso, valor_adquisicion, estado, creado_en, descripcion, factura, proveedor,
                              vida_util_anios, metodo_depreciacion
                       FROM activos""")
        rows = cur.fetchall()
    cols = ["id","nombre","categoria","ubicacion","tipo","area_responsable",
            "fecha_ingreso","valor_adquisicion","estado","creado_en","descripcion","factura","proveedor",
            "vida_util_anios","metodo_depreciacion"]
    data = [dict(zip(cols, r)) for r in rows]

    attach_ia = (include_ia or include_ia_model)
    w, feats, thr = _load_model()
    use_model = bool(w and feats) if include_ia_model else False
    today = date.today()

    for d in data:
        # --- IA (si la piden) ---
        proba_for_maint = None
        if attach_ia:
            if use_model:
                x = [1.0] + _vector_to_list(_feature_vector(d), feats)
                p = _predict_proba(w, x); t = thr; src = "model"
            else:
                p = _rule_score(d); t = _get_ai_threshold_default(); src = "rules"
            d["ia_score"] = p
            d["ia_label"] = 1 if p >= t else 0
            d["ia_source"] = src
            proba_for_maint = float(p)
        else:
            proba_for_maint = float(_rule_score(d))

        # --- Mantenimiento (si lo piden) ---
        if include_maintenance:
            edad = _years_since(d.get("fecha_ingreso") or "")
            vida = float(d.get("vida_util_anios") or 0.0)
            age_frac = min(1.0, (edad / vida)) if vida > 0 else 0.0
            base_meses = _base_interval_months(d.get("categoria"), d.get("tipo"))

            factor = 1.0 - 0.6*proba_for_maint - 0.3*age_frac
            if not math.isfinite(factor):
                factor = 1.0
            factor = max(0.3, min(1.2, factor))
            interval_meses = max(1, int(round(base_meses * factor)))

            ref_dt = _dt_from_txt(d.get("fecha_ingreso")) or _dt_from_txt(d.get("creado_en")) or datetime.utcnow()
            ref = ref_dt.date()
            if ref >= today:
                next_due = _add_months(ref, interval_meses)
            else:
                n = _months_between(ref, today)
                k = (n // interval_meses) + 1
                next_due = _add_months(ref, k * interval_meses)

            status, severity, days_to_due = _status_from_due(next_due, today, soon_days)

            d["intervalo_mantenimiento_meses"] = interval_meses
            d["proxima_fecha_mantenimiento"] = next_due.isoformat()
            d["days_to_due"] = days_to_due
            d["maint_status"] = status          # red | yellow | green
            d["maint_severity"] = severity      # 0 | 1 | 2

    # ===== KPIs y agregados (igual que tenías) =====
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

    # Limpiar campos sensibles
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
# NUEVO: REPORTE DE DEPRECIACIÓN (helper + endpoints)
# =========================

def _parse_fecha_to_date(fecha_txt: Optional[str]) -> Optional[date]:
    if not fecha_txt:
        return None
    try:
        return datetime.strptime(fecha_txt[:10], "%Y-%m-%d").date()
    except Exception:
        return None

def _calc_depreciacion_report():
    """Calcula el payload de depreciación (reutilizable para endpoint con token y público con API key)."""
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""
            SELECT id, nombre, categoria, estado, fecha_ingreso, valor_adquisicion,
                   vida_util_anios, metodo_depreciacion
            FROM activos
        """)
        rows = cur.fetchall()

    hoy = date.today()
    out = []
    for (aid, nombre, categoria, estado, f_ing, val, vida_util, metodo) in rows:
        vida_util = float(vida_util or 0) if vida_util is not None else 0.0
        val = float(val or 0.0)
        metodo = (metodo or "linea_recta").strip().lower()

        fi_date = _parse_fecha_to_date(f_ing)
        anios = max(0.0, (hoy - fi_date).days / 365.25) if fi_date else 0.0

        # Por ahora: línea recta
        gasto_anual = (val / vida_util) if vida_util > 0 else 0.0
        dep_acum = min(val, gasto_anual * anios)
        valor_libros = max(0.0, val - dep_acum)
        vida_restante = max(0.0, vida_util - anios)

        out.append({
            "id": aid,
            "nombre": nombre,
            "categoria": categoria,
            "estado": estado,
            "fecha_ingreso": f_ing,
            "valor_adquisicion": round(val, 2),
            "vida_util_anios": round(vida_util, 2),
            "metodo_depreciacion": metodo,
            "anios_transcurridos": round(anios, 2),
            "depreciacion_anual": round(gasto_anual, 2),
            "depreciacion_acumulada": round(dep_acum, 2),
            "valor_en_libros": round(valor_libros, 2),
            "vida_util_restante": round(vida_restante, 2),
        })
    return {"detalle": out, "total": len(out), "generado_en": datetime.utcnow().isoformat()}

# Endpoint con token (interno)
@app.get("/reportes/activos_depreciacion")
def reporte_activos_depreciacion(user=Depends(get_current_user)):
    return _calc_depreciacion_report()

# Endpoint público para Power BI (API key por querystring)
@app.get("/export/public/activos_depreciacion.json")
def export_depreciacion_public(request: Request):
    check_export_key(request)  # usa ?key=TU_API_KEY
    return _calc_depreciacion_report()


# =========================
# DOCUMENTOS PDF
# =========================
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF

# ---------- Helpers generales ----------
def _fetch_asset(asset_id: int):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(_COMMON_SELECT + " WHERE id=?", (asset_id,))
        r = cur.fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Activo no encontrado")
    cols = ["id","nombre","descripcion","categoria","ubicacion","factura","proveedor",
            "tipo","area_responsable","fecha_ingreso","valor_adquisicion","estado","creado_en",
            "vida_util_anios","metodo_depreciacion"]
    return dict(zip(cols, r))

def _qr_drawing(text: str, size: float = 2.4 * cm) -> Drawing:
    code = qr.QrCodeWidget(text)
    b = code.getBounds()
    w = b[2] - b[0]
    h = b[3] - b[1]
    k = min(size / w, size / h)
    d = Drawing(size, size, transform=[k, 0, 0, k, 0, 0])
    d.add(code)
    return d

def _hcell(txt):   # header cell style helper
    return Paragraph(f"<b>{txt}</b>", getSampleStyleSheet()["BodyText"])

def _small(txt):
    st = ParagraphStyle("small", fontSize=8, leading=10)
    return Paragraph(str(txt or ""), st)

def _p(txt):
    return Paragraph(str(txt or ""), getSampleStyleSheet()["BodyText"])

def _mk_doc(buf):
    return SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.3*cm, rightMargin=1.3*cm, topMargin=1.3*cm, bottomMargin=1.3*cm
    )

def _line():
    return Table([[""]], colWidths=[16*cm], style=[("LINEABOVE", (0,0), (-1,-1), 0.5, colors.grey)])

# ---------- Plantilla: Comprobante de Ingreso ----------
def _pdf_comprobante_ingreso(a: dict, logo_path: str | None = None):
    buf = io.BytesIO()
    doc = _mk_doc(buf)
    S = getSampleStyleSheet()
    story = []

    # Encabezado
    header_tbl = [
        [
            Image(logo_path, width=2.8*cm, height=1.4*cm) if logo_path else "",
            Paragraph("<b>DEPARTAMENTO FINANCIERO – CONTROL DE BIENES</b><br/>"
                      "<b>COMPROBANTE DE INGRESO DE ACTIVOS FIJOS</b>", S["Title"]),
            _qr_drawing(f"ingreso:{a['id']}|{a.get('nombre','')}|{a.get('fecha_ingreso','')}")
        ]
    ]
    story.append(Table(header_tbl, colWidths=[3*cm, 10*cm, 3*cm], style=[
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")
    ]))
    story.append(Spacer(1, 0.25*cm))

    # Datos generales
    hoy = datetime.utcnow().strftime("%Y-%m-%d")
    info = [
        [_hcell("Fecha"), _p(hoy), _hcell("Proveedor"), _p(a.get("proveedor",""))],
        [_hcell("Custodio"), _p(a.get("area_responsable","")), _hcell("Factura N°"), _p(a.get("factura",""))],
        [_hcell("Departamento"), _p(a.get("ubicacion","")), _hcell("Fecha de Factura"), _p(a.get("fecha_ingreso",""))],
        [_hcell("Tipo de Proceso"), _p(a.get("metodo_depreciacion","")), _hcell("Estado"), _p(a.get("estado",""))],
    ]
    story.append(Table(info, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm], style=[
        ("BOX",(0,0),(-1,-1),0.6,colors.grey),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(Spacer(1, 0.35*cm))

    # Detalle del bien
    detalle_header = [["CÓDIGO", "DESCRIPCIÓN DEL BIEN", "CATEGORÍA", "TIPO", "VALOR (USD)"]]
    detalle_row = [[
        f"AF_{str(a['id']).zfill(5)}",
        f"{a.get('nombre','')}: {a.get('descripcion','')}",
        a.get("categoria",""), a.get("tipo",""),
        f"{(a.get('valor_adquisicion') or 0):,.2f}"
    ]]
    story.append(Table(detalle_header + detalle_row, colWidths=[3*cm, 7*cm, 2.5*cm, 2.5*cm, 3*cm], style=[
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("BOX",(0,0),(-1,-1),0.6,colors.grey),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
        ("ALIGN",(-1,1),(-1,1),"RIGHT"),
    ]))
    story.append(Spacer(1, 0.2*cm))

    # Observaciones
    story.append(_hcell("Observaciones"))
    story.append(Paragraph("EL EQUIPO FUE REVISADO Y CODIFICADO EN LA BODEGA. "
                           "SE ENCUENTRA EN BUEN ESTADO.", S["BodyText"]))
    story.append(Spacer(1, 0.4*cm))

    # Firmas
    firmas = [
        [_small("_______________________________"), "", _small("_______________________________")],
        [_small("DIRECTOR FINANCIERO"), "", _small("CUSTODIO ENTRANTE")],
    ]
    story.append(Table(firmas, colWidths=[6.5*cm, 3*cm, 6.5*cm], style=[("ALIGN",(0,0),(-1,-1),"CENTER")]))

    doc.build(story)
    buf.seek(0)
    return buf

# ---------- Plantilla: Acta (Entrega/Baja) ----------
def _pdf_acta_generic(a: dict, titulo: str, observacion: str = "", logo_path: str | None = None):
    buf = io.BytesIO()
    doc = _mk_doc(buf)
    S = getSampleStyleSheet()
    story = []

    # Encabezado
    story.append(Table([[
        Image(logo_path, width=2.8*cm, height=1.4*cm) if logo_path else "",
        Paragraph(f"<b>ACTA DE {titulo.upper()}</b>", S["Title"]),
        _qr_drawing(f"{titulo.lower()}:{a['id']}|{a.get('nombre','')}")
    ]], colWidths=[3*cm, 10*cm, 3*cm], style=[("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(Spacer(1, 0.25*cm))

    # Datos generales
    hoy = datetime.utcnow().strftime("%Y-%m-%d")
    datos = [
        [_hcell("Fecha"), _p(hoy), _hcell("Departamento / Área"), _p(a.get("ubicacion",""))],
        [_hcell("Custodio/Responsable"), _p(a.get("area_responsable","")), _hcell("Código"), _p(f"AF_{str(a['id']).zfill(5)}")],
    ]
    story.append(Table(datos, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm], style=[
        ("BOX",(0,0),(-1,-1),0.6,colors.grey),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(Spacer(1, 0.3*cm))

    # Tabla de artículo
    header = [["UNIDAD DE MEDIDA", "DESCRIPCIÓN DEL BIEN / ARTÍCULO", "ESTADO", "CANTIDAD"]]
    row = [["UNIDAD",
            f"{a.get('nombre','')} — {a.get('descripcion','')}",
            (a.get("estado") or "NUEVO").upper(),
            "1"]]
    story.append(Table(header + row, colWidths=[3.5*cm, 8.5*cm, 2*cm, 2*cm], style=[
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("BOX",(0,0),(-1,-1),0.6,colors.grey),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
        ("ALIGN",(-1,1),(-1,1),"CENTER"),
    ]))
    story.append(Spacer(1, 0.2*cm))

    # Observaciones
    story.append(_hcell("Observación"))
    story.append(Paragraph(observacion or "SE ENTREGA EL BIEN EN EL ESTADO QUE SE INDICA.", S["BodyText"]))
    story.append(Spacer(1, 0.45*cm))

    # Firmas (Solicita / Entrega / Recibe)
    firmas = [
        [_small("_______________________________"), _small("_______________________________"), _small("_______________________________")],
        [_small("SOLICITADO POR"), _small("ENTREGA"), _small("RECIBE CONFORME")],
        [_small("Cargo: _______________________"), _small("Cargo: _______________________"), _small("Cargo: _______________________")],
    ]
    story.append(Table(firmas, colWidths=[5.3*cm, 5.3*cm, 5.3*cm], style=[("ALIGN",(0,0),(-1,-1),"CENTER")]))

    doc.build(story)
    buf.seek(0)
    return buf

# ---------- Endpoints de documentos ----------
@app.get("/documentos/comprobante/ingreso/{asset_id}.pdf")
def comprobante_ingreso(asset_id: int, user=Depends(get_current_user)):
    a = _fetch_asset(asset_id)
    pdf = _pdf_comprobante_ingreso(a)  # logo opcional: _pdf_comprobante_ingreso(a, logo_path="static/logo.png")
    return StreamingResponse(pdf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="comprobante_ingreso_{asset_id}.pdf"'})

class ActaEntregaBody(BaseModel):
    observacion: Optional[str] = "SE ENTREGA EL BIEN EN ESTADO NUEVO"

@app.post("/documentos/acta/entrega/{asset_id}.pdf")
def acta_entrega(asset_id: int, body: ActaEntregaBody, user=Depends(get_current_user)):
    a = _fetch_asset(asset_id)
    pdf = _pdf_acta_generic(a, titulo="ENTREGA/RECEPCIÓN", observacion=body.observacion)
    return StreamingResponse(pdf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="acta_entrega_{asset_id}.pdf"'})

class BajaRequest(BaseModel):
    motivo: str = "BAJA POR OBSOLESCENCIA"

@app.post("/documentos/acta/baja/{asset_id}.pdf")
def acta_baja(asset_id: int, req: BajaRequest, user=Depends(require_role(["supervisor","management"]))):
    # Cambia estado a baja
    with get_conn() as c:
        c.execute("UPDATE activos SET estado=? WHERE id=?", ("baja", asset_id))
    a = _fetch_asset(asset_id)
    obs = f"MOTIVO DE BAJA: {req.motivo}"
    pdf = _pdf_acta_generic(a, titulo="BAJA", observacion=obs)
    return StreamingResponse(pdf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="acta_baja_{asset_id}.pdf"'})
# ===== LOTE: ACTA DE ENTREGA (multi-activos en un solo PDF)
class ActaEntregaLoteBody(BaseModel):
    asset_ids: List[int]
    observacion: Optional[str] = "SE ENTREGA EL BIEN EN EL ESTADO QUE SE INDICA."

@app.post("/documentos/acta/entrega/lote.pdf")
def acta_entrega_lote(body: ActaEntregaLoteBody, user=Depends(get_current_user)):
    if not body.asset_ids:
        raise HTTPException(400, "asset_ids vacío")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.3*cm, rightMargin=1.3*cm, topMargin=1.3*cm, bottomMargin=1.3*cm
    )
    S = getSampleStyleSheet()
    story = []
    for idx, aid in enumerate(body.asset_ids):
        a = _fetch_asset(aid)
        # Reutiliza la misma maquetación del acta genérica:
        # (copiamos el contenido esencial para construir cada página)
        story.append(Table([[
            "",  # logo opcional
            Paragraph(f"<b>ACTA DE ENTREGA/RECEPCIÓN</b>", S["Title"]),
            _qr_drawing(f"entrega:{a['id']}|{a.get('nombre','')}")
        ]], colWidths=[3*cm, 10*cm, 3*cm], style=[("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(Spacer(1, 0.25*cm))

        hoy = datetime.utcnow().strftime("%Y-%m-%d")
        datos = [
            [ _hcell("Fecha"), _p(hoy), _hcell("Departamento / Área"), _p(a.get("ubicacion","")) ],
            [ _hcell("Custodio/Responsable"), _p(a.get("area_responsable","")), _hcell("Código"), _p(f"AF_{str(a['id']).zfill(5)}") ],
        ]
        story.append(Table(datos, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm], style=[
            ("BOX",(0,0),(-1,-1),0.6,colors.grey),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(Spacer(1, 0.3*cm))

        header = [["UNIDAD DE MEDIDA", "DESCRIPCIÓN DEL BIEN / ARTÍCULO", "ESTADO", "CANTIDAD"]]
        row = [["UNIDAD", f"{a.get('nombre','')} — {a.get('descripcion','')}", (a.get('estado') or 'NUEVO').upper(), "1"]]
        story.append(Table(header + row, colWidths=[3.5*cm, 8.5*cm, 2*cm, 2*cm], style=[
            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
            ("BOX",(0,0),(-1,-1),0.6,colors.grey),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
            ("ALIGN",(-1,1),(-1,1),"CENTER"),
        ]))
        story.append(Spacer(1, 0.2*cm))

        story.append(_hcell("Observación"))
        story.append(Paragraph(body.observacion or "", S["BodyText"]))
        story.append(Spacer(1, 0.45*cm))

        firmas = [
            [_small("_______________________________"), _small("_______________________________"), _small("_______________________________")],
            [_small("SOLICITADO POR"), _small("ENTREGA"), _small("RECIBE CONFORME")],
            [_small("Cargo: _______________________"), _small("Cargo: _______________________"), _small("Cargo: _______________________")],
        ]
        story.append(Table(firmas, colWidths=[5.3*cm, 5.3*cm, 5.3*cm], style=[("ALIGN",(0,0),(-1,-1),"CENTER")]))

        if idx < len(body.asset_ids) - 1:
            story.append(Spacer(1, 0.1*cm))
            story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="acta_entrega_lote.pdf"'})


# ===== LOTE: ACTA DE BAJA (multi-activos)
class ActaBajaLoteBody(BaseModel):
    asset_ids: List[int]
    motivo: str = "BAJA POR OBSOLESCENCIA"

@app.post("/documentos/acta/baja/lote.pdf")
def acta_baja_lote(body: ActaBajaLoteBody, user=Depends(require_role(["supervisor","management"]))):
    if not body.asset_ids:
        raise HTTPException(400, "asset_ids vacío")

    # Opcional: cambiar estado a 'baja' de todos
    with get_conn() as c:
        c.executemany("UPDATE activos SET estado=? WHERE id=?", [("baja", aid) for aid in body.asset_ids])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.3*cm, rightMargin=1.3*cm, topMargin=1.3*cm, bottomMargin=1.3*cm
    )
    S = getSampleStyleSheet()
    story = []
    for idx, aid in enumerate(body.asset_ids):
        a = _fetch_asset(aid)

        story.append(Table([[
            "",  # logo opcional
            Paragraph("<b>ACTA DE BAJA</b>", S["Title"]),
            _qr_drawing(f"baja:{a['id']}|{a.get('nombre','')}")
        ]], colWidths=[3*cm, 10*cm, 3*cm], style=[("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(Spacer(1, 0.25*cm))

        hoy = datetime.utcnow().strftime("%Y-%m-%d")
        datos = [
            [ _hcell("Fecha"), _p(hoy), _hcell("Departamento / Área"), _p(a.get("ubicacion","")) ],
            [ _hcell("Custodio/Responsable"), _p(a.get("area_responsable","")), _hcell("Código"), _p(f"AF_{str(a['id']).zfill(5)}") ],
        ]
        story.append(Table(datos, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm], style=[
            ("BOX",(0,0),(-1,-1),0.6,colors.grey),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(Spacer(1, 0.3*cm))

        header = [["UNIDAD DE MEDIDA", "DESCRIPCIÓN DEL BIEN / ARTÍCULO", "ESTADO", "CANTIDAD"]]
        row = [["UNIDAD", f"{a.get('nombre','')} — {a.get('descripcion','')}", (a.get('estado') or 'BAJA').upper(), "1"]]
        story.append(Table(header + row, colWidths=[3.5*cm, 8.5*cm, 2*cm, 2*cm], style=[
            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
            ("BOX",(0,0),(-1,-1),0.6,colors.grey),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
            ("ALIGN",(-1,1),(-1,1),"CENTER"),
        ]))
        story.append(Spacer(1, 0.2*cm))

        story.append(_hcell("Observación"))
        story.append(Paragraph(f"MOTIVO DE BAJA: {body.motivo}", S["BodyText"]))
        story.append(Spacer(1, 0.45*cm))

        firmas = [
            [_small("_______________________________"), _small("_______________________________"), _small("_______________________________")],
            [_small("SOLICITADO POR"), _small("ENTREGA"), _small("RECIBE CONFORME")],
            [_small("Cargo: _______________________"), _small("Cargo: _______________________"), _small("Cargo: _______________________")],
        ]
        story.append(Table(firmas, colWidths=[5.3*cm, 5.3*cm, 5.3*cm], style=[("ALIGN",(0,0),(-1,-1),"CENTER")]))

        if idx < len(body.asset_ids) - 1:
            story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="acta_baja_lote.pdf"'})


# --- HELPERS DE GRÁFICAS Y PDF ---
def _fig_to_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def _barh_from_dict(d: dict, title: str, xlabel: str, ylabel: str):
    # Orden descendente por valor
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center")
    return _fig_to_png_bytes(fig)

def _sort_mes_ingreso_counter(counter: dict):
    # Ordena YYYY-MM de forma cronológica y deja "sin_fecha" (u otros) al final
    keys_valid = [k for k in counter.keys() if isinstance(k, str) and len(k) == 7 and k[4] == "-"]
    keys_valid.sort()
    ordered = {k: counter[k] for k in keys_valid}
    for k in counter.keys():
        if k not in ordered:
            ordered[k] = counter[k]
    return ordered

def _normalize_ym(fecha):
    if not fecha:
        return "sin_fecha"
    if isinstance(fecha, (datetime, date)):
        return fecha.strftime("%Y-%m")
    s = str(fecha)
    return s[:7] if len(s) >= 7 else "sin_fecha"

def _tabla_kpis(k):
    data = [
        ["Total de activos", "Activos operacionales", "Activos de baja"],
        [str(k["total_activos"]), str(k["activos_operacionales"]), str(k["activos_baja"])],
    ]
    t = Table(data, colWidths=[6*cm, 6*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,0), 11),
        ("FONTSIZE", (0,1), (-1,1), 14),
        ("BOX", (0,0), (-1,-1), 0.6, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
    ]))
    return t

def _tabla_preview(detalle, max_rows=15):
    preview_cols = ["id","nombre","categoria","estado","fecha_ingreso","valor_adquisicion"]
    data = [preview_cols]
    for d in detalle[:max_rows]:
        row = [
            str(d.get("id","")),
            str(d.get("nombre","")),
            str(d.get("categoria","")),
            str(d.get("estado","")),
            str(_normalize_ym(d.get("fecha_ingreso",""))),
            str(d.get("valor_adquisicion","")),
        ]
        data.append(row)
    t = Table(data, repeatRows=1, colWidths=[2*cm,5*cm,3*cm,2.5*cm,3*cm,3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
    ]))
    return t

def _build_pdf_ejecutivo(rep, logo_path=None):
    """Crea y devuelve BytesIO del PDF con KPIs + 3 gráficas + preview."""
    k = rep["kpis"]; agg = rep["agregados"]; detalle = rep["detalle"]

    # 1) Preparar imágenes de gráficas
    g_categoria = _barh_from_dict(agg.get("por_categoria", {}), "Recuento por categoría", "Cantidad", "Categoría")
    g_estado    = _barh_from_dict(agg.get("por_estado", {}), "Recuento por estado", "Cantidad", "Estado")
    por_mes_ord = _sort_mes_ingreso_counter(agg.get("por_mes_ingreso", {}))
    g_mes       = _barh_from_dict(por_mes_ord, "Altas por mes de ingreso", "Cantidad", "YYYY-MM")

    # 2) Construir PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.2*cm, rightMargin=1.2*cm,
        topMargin=1.2*cm, bottomMargin=1.2*cm
    )
    styles = getSampleStyleSheet()
    story = []

    # Cabecera
    story.append(Paragraph("Reporte Ejecutivo - Activos", styles["Title"]))
    story.append(Paragraph(datetime.utcnow().strftime("Fecha de emisión: %Y-%m-%d %H:%M UTC"), styles["Normal"]))
    story.append(Spacer(1, 0.4*cm))

    # Logo opcional
    if logo_path:
        try:
            story.append(Image(logo_path, width=3.0*cm, height=3.0*cm))
            story.append(Spacer(1, 0.3*cm))
        except Exception:
            pass

    # KPIs
    story.append(_tabla_kpis(k))
    story.append(Spacer(1, 0.5*cm))

    # Gráficas
    story.append(Paragraph("Distribuciones", styles["Heading2"]))
    for img_buf in (g_categoria, g_estado, g_mes):
        story.append(Image(img_buf, width=16*cm, height=7*cm))
        story.append(Spacer(1, 0.25*cm))

    # Preview de detalle
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Muestra de detalle", styles["Heading2"]))
    story.append(_tabla_preview(detalle))

    doc.build(story)
    buf.seek(0)
    return buf

# --- TU ENDPOINT (reemplaza el tuyo por este) ---
@app.get("/reportes/instantaneo.pdf")
def reporte_ejecutivo_pdf(user=Depends(get_current_user)):
    # Reutiliza tu endpoint de datos
    rep = reportes_activos(user=user)

    pdf_buf = _build_pdf_ejecutivo(rep)  # puedes pasar logo_path="static/logo.png" si quieres
    headers = {"Content-Disposition": 'attachment; filename="reporte_ejecutivo.pdf"'}
    return StreamingResponse(pdf_buf, media_type="application/pdf", headers=headers)
# =========================
# FIN
# =========================
