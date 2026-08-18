import os
import re
import time
import hmac
import hashlib
import asyncio
import sqlite3
import requests
import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional PostgreSQL Driver (Supabase Pooler)
try:
    import psycopg2
    from psycopg2 import pool
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

app = FastAPI(
    title="SheetPulse AI Enterprise Core",
    version="26.0.0",
    docs_url="/api/swagger",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Cluster API Keys ---
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_SheetPulseDemo")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "demo_secret_key")
SQLITE_DB_PATH = os.getenv("SHEETPULSE_DB_PATH", "sheetpulse.db")

IS_POSTGRES = HAS_POSTGRES and (DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://"))
pg_pool = None

if IS_POSTGRES:
    cleaned_url = DATABASE_URL
    if cleaned_url.startswith("postgres://"):
        cleaned_url = cleaned_url.replace("postgres://", "postgresql://", 1)
    try:
        pg_pool = psycopg2.pool.SimpleConnectionPool(1, 20, cleaned_url, sslmode="require")
    except Exception as e:
        print(f"Warning: Supabase connection failed ({e}), falling back to SQLite.")
        IS_POSTGRES = False

class DBConn:
    def __init__(self):
        self.is_pg = IS_POSTGRES
        self.conn = None

    def __enter__(self):
        if self.is_pg:
            self.conn = pg_pool.getconn()
        else:
            self.conn = sqlite3.connect(SQLITE_DB_PATH, timeout=25.0)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_pg:
            if self.conn:
                if exc_type:
                    self.conn.rollback()
                else:
                    self.conn.commit()
                pg_pool.putconn(self.conn)
        else:
            if self.conn:
                if exc_type:
                    self.conn.rollback()
                else:
                    self.conn.commit()
                self.conn.close()

    def execute(self, query: str, params: tuple = ()):
        cur = self.conn.cursor()
        if self.is_pg:
            formatted_query = query.replace("?", "%s")
            cur.execute(formatted_query, params)
        else:
            cur.execute(query, params)
        return cur

def init_db():
    try:
        with DBConn() as db:
            if db.is_pg:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key TEXT PRIMARY KEY,
                        owner_name TEXT NOT NULL,
                        tier TEXT DEFAULT 'free',
                        credits_left INTEGER DEFAULT 100,
                        total_used INTEGER DEFAULT 0,
                        created_at DOUBLE PRECISION NOT NULL
                    )
                """)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        payment_id TEXT,
                        owner_name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        tier TEXT NOT NULL,
                        amount INTEGER NOT NULL,
                        status TEXT DEFAULT 'created',
                        created_at DOUBLE PRECISION NOT NULL
                    )
                """)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS usage_logs (
                        id SERIAL PRIMARY KEY,
                        key TEXT,
                        action TEXT,
                        provider TEXT,
                        latency REAL,
                        input_length INTEGER,
                        timestamp DOUBLE PRECISION NOT NULL
                    )
                """)
            else:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key TEXT PRIMARY KEY,
                        owner_name TEXT NOT NULL,
                        tier TEXT DEFAULT 'free',
                        credits_left INTEGER DEFAULT 100,
                        total_used INTEGER DEFAULT 0,
                        created_at REAL NOT NULL
                    )
                """)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        payment_id TEXT,
                        owner_name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        tier TEXT NOT NULL,
                        amount INTEGER NOT NULL,
                        status TEXT DEFAULT 'created',
                        created_at REAL NOT NULL
                    )
                """)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS usage_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT,
                        action TEXT,
                        provider TEXT,
                        latency REAL,
                        input_length INTEGER,
                        timestamp REAL NOT NULL
                    )
                """)

            cur = db.execute("SELECT key FROM api_keys WHERE key = ?", ("sp_demo_live",))
            if not cur.fetchone():
                db.execute(
                    "INSERT INTO api_keys VALUES (?, ?, 'free', 10000, 0, ?)",
                    ("sp_demo_live", "Web Playground Sandbox", time.time())
                )
    except Exception as e:
        print(f"Database Init Notice: {e}")

init_db()

def is_byok_key(key: str) -> bool:
    k = (key or "").strip()
    return k.startswith("gsk_") or k.startswith("csk-") or k.startswith("sk-") or k.startswith("AIza")

def verify_and_deduct_credits(api_key: str, amount: int = 1) -> Dict[str, Any]:
    sanitized_key = (api_key or "").strip()

    if not sanitized_key or sanitized_key in ["sp_demo_live", "demo", "sandbox", "YOUR_API_KEY_HERE"]:
        return {"owner": "Web Sandbox Demo", "tier": "free", "remaining": 9999}

    with DBConn() as db:
        cur = db.execute("SELECT owner_name, tier, credits_left, total_used FROM api_keys WHERE key = ?", (sanitized_key,))
        row = cur.fetchone()

        if not row and is_byok_key(sanitized_key):
            provider_tag = "Groq BYOK" if sanitized_key.startswith("gsk_") else ("Cerebras BYOK" if sanitized_key.startswith("csk-") else "BYOK Provider")
            db.execute(
                "INSERT INTO api_keys VALUES (?, ?, 'byok', 999999, ?, ?)",
                (sanitized_key, provider_tag, amount, time.time())
            )
            return {"owner": provider_tag, "tier": "BYOK", "remaining": 999999}

        if not row and sanitized_key.startswith("sp_"):
            db.execute(
                "INSERT INTO api_keys VALUES (?, ?, 'free', 100, ?, ?)",
                (sanitized_key, "Workspace User", amount, time.time())
            )
            return {"owner": "Workspace User", "tier": "free", "remaining": 100 - amount}

        if not row:
            raise HTTPException(status_code=401, detail="Invalid API Key. Please generate a valid key.")

        owner_name, tier, credits_left, total_used = row[0], row[1], row[2], row[3]
        if tier not in ["developer", "byok"] and credits_left < amount:
            raise HTTPException(status_code=402, detail="Credit quota exhausted. Please upgrade.")

        if tier == "byok":
            db.execute("UPDATE api_keys SET total_used = total_used + ? WHERE key = ?", (amount, sanitized_key))
        else:
            db.execute("UPDATE api_keys SET credits_left = credits_left - ?, total_used = total_used + ? WHERE key = ?", (amount, amount, sanitized_key))

        return {"owner": owner_name, "tier": tier, "remaining": credits_left if tier == "byok" else credits_left - amount}

def log_request_event(api_key: str, action: str, provider: str, latency: float, input_len: int):
    try:
        with DBConn() as db:
            db.execute(
                "INSERT INTO usage_logs (key, action, provider, latency, input_length, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (api_key or "sp_demo_live", action, provider, latency, input_len, time.time())
            )
    except Exception:
        pass

MEMORY_CACHE: Dict[str, dict] = {}
MAX_CACHE_ENTRIES = 5000
CONCURRENCY_SEMAPHORE = asyncio.Semaphore(35)

REGEX_PATTERNS = {
    "email": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    "phone": r'(\+?[0-9]{1,3}[-.\s]?)?\(?[0-9]{2,5}\)?[-.\s]?[0-9]{3,5}[-.\s]?[0-9]{3,5}',
    "url": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
    "price": r'[\$\€\£\₹]\s?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?',
    "date": r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
    "number": r'\b\d+(?:\.\d+)?\b'
}

def try_fast_regex_extract(text: str, target: str) -> Optional[str]:
    target_lower = (target or "").lower().strip()
    for key, pattern in REGEX_PATTERNS.items():
        if key in target_lower:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
    return None

def fetch_url_text(url: str) -> str:
    try:
        clean_url = url.strip().strip('"\'')
        if not clean_url.startswith(('http://', 'https://')):
            clean_url = 'https://' + clean_url
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        res = requests.get(clean_url, headers=headers, timeout=8)
        html = res.text
        text = re.sub(r'<script.*?</script>|<style.*?</style>|<header.*?</header>|<footer.*?</footer>|<nav.*?</nav>', '', html, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = ' '.join(text.split())
        return clean_text[:3500] if clean_text else "Empty web content."
    except Exception as e:
        return f"Scrape Notice: ({str(e)})"

def clean_output(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)

    final_patterns = [
        r'\*\*Final Answer:?\*\*\s*(.+)',
        r'Final Answer:?\s*(.+)',
        r'\*\*Answer:?\*\*\s*(.+)',
        r'Answer:?\s*(.+)'
    ]
    for pattern in final_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1)
            break

    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    lines = [
        l.strip() for l in text.splitlines() 
        if l.strip() and not l.strip().startswith(('---', '###', '|', 'Option', 'Why this', 'Here is', 'Sure!'))
    ]
    if lines:
        text = lines[0]
    return text.strip('*_ `').strip()

class KeyGenRequest(BaseModel):
    owner_name: str = Field(..., min_length=1, max_length=100)
    tier: Optional[str] = Field("free", pattern="^(free|pro|developer)$")

class CreateOrderRequest(BaseModel):
    owner_name: str
    email: str
    tier: str = Field("pro", pattern="^(pro|agency)$")

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    owner_name: str
    tier: str

class ProcessRequest(BaseModel):
    text: str = Field("", max_length=25000)
    instruction: Optional[str] = Field("", max_length=2000)
    action: str = Field("custom", max_length=50)
    api_key: Optional[str] = Field("sp_demo_live")

class BatchRequest(BaseModel):
    items: List[ProcessRequest] = Field(..., max_length=100)
    api_key: Optional[str] = Field("")

# ================= 3 HARDENED AI ENGINE ADAPTERS =================

# 1. Cerebras Hardware Cloud
def _sync_cerebras_call(sys_prompt: str, usr_prompt: str, custom_key: Optional[str] = None) -> Tuple[str, str]:
    key = custom_key if (custom_key and custom_key.startswith("csk-")) else CEREBRAS_API_KEY
    if not key:
        raise ValueError("Cerebras API key not set in environment")
    
    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    last_err = ""
    for model in ["llama3.1-8b", "llama-3.3-70b", "llama3.1-70b"]:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
                "temperature": 0.05,
                "max_tokens": 150
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                res = resp.json()
                out = clean_output(res["choices"][0]["message"]["content"])
                if out:
                    return out, f"Cerebras:{model}"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            last_err = str(e)
    raise ValueError(f"Cerebras execution failed -> {last_err}")

# 2. Groq Hardware Cloud
def _sync_groq_call(sys_prompt: str, usr_prompt: str, custom_key: Optional[str] = None) -> Tuple[str, str]:
    key = custom_key if (custom_key and custom_key.startswith("gsk_")) else GROQ_API_KEY
    if not key:
        raise ValueError("Groq API key not set in environment")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    last_err = ""
    for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
                "temperature": 0.05,
                "max_tokens": 150
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                res = resp.json()
                out = clean_output(res["choices"][0]["message"]["content"])
                if out:
                    return out, f"Groq:{model}"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            last_err = str(e)
    raise ValueError(f"Groq execution failed -> {last_err}")

# 3. OpenRouter Cloud Pool
def _sync_openrouter_call(sys_prompt: str, usr_prompt: str, custom_key: Optional[str] = None) -> Tuple[str, str]:
    key = custom_key if (custom_key and (custom_key.startswith("sk-or-") or custom_key.startswith("sk-"))) else OPENROUTER_API_KEY
    if not key:
        raise ValueError("OpenRouter API key not set in environment")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sheetpulseai.onrender.com",
        "X-Title": "SheetPulse AI"
    }
    last_err = ""
    for model in ["meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.0-flash-exp:free", "qwen/qwen-2.5-72b-instruct"]:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
                "temperature": 0.05,
                "max_tokens": 150
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=9)
            if resp.status_code == 200:
                res = resp.json()
                out = clean_output(res["choices"][0]["message"]["content"])
                if out:
                    return out, f"OpenRouter:{model.split('/')[-1]}"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            last_err = str(e)
    raise ValueError(f"OpenRouter execution failed -> {last_err}")

def resolve_action_prompts(action: str, instruction: str, text: str) -> Tuple[str, str]:
    act = (action or "custom").lower().strip()
    base_rule = "You are an autonomous spreadsheet formula engine. Output ONLY the concise final answer that fits in a single spreadsheet cell. ZERO conversational fluff, ZERO explanations, ZERO markdown headings, ZERO preamble."
    
    if act == "clean":
        sys = f"{base_rule} Standardize casing, remove extra whitespace, format phone/emails. Output ONLY cleaned data."
        usr = f"Input: {text}"
    elif act == "extract":
        sys = f"{base_rule} Extract requested entity. Output ONLY extracted text."
        usr = f"Target: {instruction}\nContent: {text}"
    elif act == "classify":
        sys = f"{base_rule} Classify strictly into ONE tag from: [{instruction}]. Output ONLY exact tag string."
        usr = f"Input: {text}"
    elif act == "fix_formula":
        sys = f"{base_rule} Fix broken formula. Output ONLY working formula starting with '='."
        usr = f"Broken Formula: {text}\nGoal: {instruction}"
    elif act == "scrape":
        scraped_data = fetch_url_text(text)
        sys = f"{base_rule} Extract answer from web text."
        usr = f"Question: {instruction}\nContent:\n{scraped_data}"
    else:
        sys = f"{base_rule} Execute instruction directly and return ONLY final result."
        usr = f"Instruction: {instruction}\nContext: {text}"
    return sys, usr

@app.get("/")
def serve_home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>SheetPulse AI Backend Online</h1>")

@app.get("/docs")
def serve_docs():
    if os.path.exists("docs.html"):
        return FileResponse("docs.html")
    return HTMLResponse("<h1>SheetPulse AI Documentation</h1>")

@app.get("/api/v1/health")
def health_metrics():
    with DBConn() as db:
        cur = db.execute("SELECT COUNT(*), SUM(total_used) FROM api_keys")
        u_row = cur.fetchone()
        u_count = u_row[0] if u_row else 0
        total_exec = u_row[1] if (u_row and u_row[1]) else 0

        cur_logs = db.execute("SELECT COUNT(*) FROM usage_logs")
        log_row = cur_logs.fetchone()
        log_count = log_row[0] if log_row else 0

    return {
        "status": "online",
        "service": "SheetPulse AI Enterprise Core",
        "version": "26.0.0",
        "database": "Supabase (PostgreSQL)" if IS_POSTGRES else "Local (SQLite)",
        "active_keys": u_count,
        "total_cells_processed": total_exec,
        "logged_events": log_count,
        "cluster_providers": {
            "cerebras": bool(CEREBRAS_API_KEY),
            "groq": bool(GROQ_API_KEY),
            "openrouter": bool(OPENROUTER_API_KEY)
        }
    }

# --- ISOLATED RAW DIAGNOSTIC ENDPOINT ---
@app.get("/api/v1/debug/providers")
def debug_individual_providers():
    test_sys = "Output ONLY the word 'OK'."
    test_usr = "Status check."
    results = {}

    # Test Cerebras
    try:
        out, prov = _sync_cerebras_call(test_sys, test_usr)
        results["cerebras"] = {"status": "success", "provider": prov, "output": out}
    except Exception as e:
        results["cerebras"] = {"status": "failed", "error": str(e)}

    # Test Groq
    try:
        out, prov = _sync_groq_call(test_sys, test_usr)
        results["groq"] = {"status": "success", "provider": prov, "output": out}
    except Exception as e:
        results["groq"] = {"status": "failed", "error": str(e)}

    # Test OpenRouter
    try:
        out, prov = _sync_openrouter_call(test_sys, test_usr)
        results["openrouter"] = {"status": "success", "provider": prov, "output": out}
    except Exception as e:
        results["openrouter"] = {"status": "failed", "error": str(e)}

    return {"diagnostic_report": results}

@app.get("/api/v1/dashboard/stats")
def get_user_dashboard(api_key: str):
    sanitized_key = (api_key or "").strip()
    if not sanitized_key:
        raise HTTPException(status_code=400, detail="API Key parameter required")

    with DBConn() as db:
        cur = db.execute("SELECT owner_name, tier, credits_left, total_used, created_at FROM api_keys WHERE key = ?", (sanitized_key,))
        key_row = cur.fetchone()

        if not key_row:
            if is_byok_key(sanitized_key):
                provider_tag = "Groq BYOK" if sanitized_key.startswith("gsk_") else ("Cerebras BYOK" if sanitized_key.startswith("csk-") else "BYOK Provider")
                db.execute("INSERT INTO api_keys VALUES (?, ?, 'byok', 999999, 0, ?)", (sanitized_key, provider_tag, time.time()))
            elif sanitized_key.startswith("sp_"):
                db.execute("INSERT INTO api_keys VALUES (?, ?, 'free', 100, 0, ?)", (sanitized_key, "Workspace User", time.time()))
            else:
                raise HTTPException(status_code=404, detail="API Key not found.")
            
            cur = db.execute("SELECT owner_name, tier, credits_left, total_used, created_at FROM api_keys WHERE key = ?", (sanitized_key,))
            key_row = cur.fetchone()

        owner_name, tier, credits_left, total_used, created_at = key_row[0], key_row[1], key_row[2], key_row[3], key_row[4]

        cur_logs = db.execute(
            "SELECT action, provider, latency, timestamp FROM usage_logs WHERE key = ? ORDER BY id DESC LIMIT 15",
            (sanitized_key,)
        )
        recent_logs = [{"action": r[0], "provider": r[1], "latency": r[2], "timestamp": r[3]} for r in cur_logs.fetchall()]

        cur_avg = db.execute("SELECT COUNT(*), AVG(latency) FROM usage_logs WHERE key = ?", (sanitized_key,))
        avg_row = cur_avg.fetchone()
        log_count = avg_row[0] if avg_row else 0
        avg_lat = (avg_row[1] if (avg_row and avg_row[1]) else 0.22)

        actual_total = max(total_used, log_count)

    is_unlimited = (tier == "byok")
    used_pct = 0 if is_unlimited else min(100, round((actual_total / max(1, actual_total + credits_left)) * 100))

    return {
        "success": True,
        "owner": owner_name,
        "tier": "UNLIMITED BYOK" if is_unlimited else tier.upper(),
        "credits_left": "Unlimited (BYOK)" if is_unlimited else credits_left,
        "total_used": actual_total,
        "used_percentage": used_pct,
        "avg_latency": f"{avg_lat:.2f}s",
        "created_at": time.strftime("%d %b %Y", time.localtime(created_at)),
        "recent_logs": recent_logs
    }

@app.post("/api/v1/payments/create-order")
def create_payment_order(req: CreateOrderRequest):
    amount_in_paise = 99900 if req.tier == "pro" else 249900
    generated_order_id = f"order_{uuid.uuid4().hex[:14]}"
    
    with DBConn() as db:
        db.execute(
            "INSERT INTO orders (order_id, owner_name, email, tier, amount, status, created_at) VALUES (?, ?, ?, ?, ?, 'created', ?)",
            (generated_order_id, req.owner_name.strip(), req.email.strip(), req.tier, amount_in_paise, time.time())
        )

    return {
        "success": True,
        "order_id": generated_order_id,
        "amount": amount_in_paise,
        "currency": "INR",
        "key_id": RAZORPAY_KEY_ID,
        "tier": req.tier,
        "name": req.owner_name,
        "email": req.email
    }

@app.post("/api/v1/payments/verify-payment")
def verify_payment(req: VerifyPaymentRequest):
    if RAZORPAY_KEY_SECRET != "demo_secret_key":
        generated_signature = hmac.new(
            RAZORPAY_KEY_SECRET.encode('utf-8'),
            f"{req.razorpay_order_id}|{req.razorpay_payment_id}".encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(generated_signature, req.razorpay_signature):
            raise HTTPException(status_code=400, detail="Invalid Payment Signature")

    new_key = f"sp_{uuid.uuid4().hex[:18]}"
    credits_allotted = 5000 if req.tier == "pro" else 30000

    with DBConn() as db:
        db.execute(
            "UPDATE orders SET payment_id = ?, status = 'paid' WHERE order_id = ?",
            (req.razorpay_payment_id, req.razorpay_order_id)
        )
        db.execute(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, 0, ?)",
            (new_key, req.owner_name.strip(), req.tier, credits_allotted, time.time())
        )

    return {
        "success": True,
        "api_key": new_key,
        "owner": req.owner_name,
        "credits": credits_allotted,
        "tier": req.tier.upper(),
        "payment_id": req.razorpay_payment_id
    }

@app.post("/api/v1/keys/new")
def create_free_api_key(req: KeyGenRequest):
    new_key = f"sp_{uuid.uuid4().hex[:18]}"
    initial_credits = 100 if req.tier == "free" else 5000
    with DBConn() as db:
        db.execute(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, 0, ?)", 
            (new_key, req.owner_name.strip(), req.tier, initial_credits, time.time())
        )
    return {"success": True, "api_key": new_key, "owner": req.owner_name, "credits": initial_credits, "tier": req.tier}

# --- PROCESS CELL PIPELINE ---
@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    start_time = time.time()
    text_content = (req.text or "").strip()
    if not text_content:
        return {"success": True, "result": "", "cached": False, "provider": "None"}

    effective_key = (req.api_key or "").strip()
    if not effective_key or effective_key == "YOUR_API_KEY_HERE":
        effective_key = "sp_demo_live"

    verify_and_deduct_credits(effective_key, amount=1)

    if req.action == "extract" and req.instruction:
        fast_match = try_fast_regex_extract(text_content, req.instruction)
        if fast_match:
            elapsed = round(time.time() - start_time, 3)
            log_request_event(effective_key, "extract", "Regex:UltraFast", elapsed, len(text_content))
            return {"success": True, "result": fast_match, "provider": "Regex:UltraFast", "cached": False}

    cache_key = hashlib.sha256(f"{req.action}:{req.instruction}:{text_content}".lower().encode()).hexdigest()
    if cache_key in MEMORY_CACHE:
        elapsed = round(time.time() - start_time, 3)
        log_request_event(effective_key, req.action, "Memory:Cache", elapsed, len(text_content))
        return {
            "success": True,
            "result": MEMORY_CACHE[cache_key]["result"],
            "provider": MEMORY_CACHE[cache_key]["provider"],
            "cached": True
        }

    sys_prompt, usr_prompt = resolve_action_prompts(req.action, req.instruction or "", text_content)

    async with CONCURRENCY_SEMAPHORE:
        result, provider = None, None
        
        # Priority BYOK Keys
        if effective_key.startswith("csk-"):
            try:
                result, provider = await asyncio.to_thread(_sync_cerebras_call, sys_prompt, usr_prompt, effective_key)
            except Exception:
                pass
        elif effective_key.startswith("gsk_"):
            try:
                result, provider = await asyncio.to_thread(_sync_groq_call, sys_prompt, usr_prompt, effective_key)
            except Exception:
                pass
        elif effective_key.startswith("sk-"):
            try:
                result, provider = await asyncio.to_thread(_sync_openrouter_call, sys_prompt, usr_prompt, effective_key)
            except Exception:
                pass

        # Action-Based Dynamic Routing
        if not result:
            act = req.action.lower()
            if act in ["clean", "extract"]:
                pipeline = [_sync_cerebras_call, _sync_groq_call, _sync_openrouter_call]
            elif act in ["classify", "fix_formula"]:
                pipeline = [_sync_groq_call, _sync_cerebras_call, _sync_openrouter_call]
            else:
                pipeline = [_sync_openrouter_call, _sync_groq_call, _sync_cerebras_call]

            for engine_func in pipeline:
                try:
                    result, provider = await asyncio.to_thread(engine_func, sys_prompt, usr_prompt)
                    if result:
                        break
                except Exception:
                    continue

        if not result:
            raise HTTPException(status_code=503, detail="All 3 cluster inference engines busy. Please retry.")

        if len(MEMORY_CACHE) >= MAX_CACHE_ENTRIES:
            MEMORY_CACHE.pop(next(iter(MEMORY_CACHE)))
        MEMORY_CACHE[cache_key] = {"result": result, "provider": provider}

        elapsed = round(time.time() - start_time, 3)
        log_request_event(effective_key, req.action, provider, elapsed, len(text_content))

        return {"success": True, "result": result, "provider": provider, "cached": False}

@app.post("/api/v1/batch")
async def process_batch(batch: BatchRequest):
    async def worker(item: ProcessRequest):
        try:
            item.api_key = batch.api_key
            return await process_cell(item)
        except HTTPException as he:
            return {"success": False, "error": he.detail, "status_code": he.status_code}
        except Exception as e:
            return {"success": False, "error": str(e)}

    results = await asyncio.gather(*[worker(item) for item in batch.items])
    return {"success": True, "processed_count": len(results), "data": results}
