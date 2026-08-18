import os
import re
import time
import hmac
import hashlib
import asyncio
import sqlite3
import urllib.request
import urllib.error
import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

app = FastAPI(
    title="SheetPulse AI Enterprise Core",
    version="16.0.0",
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_SheetPulseDemo")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "demo_secret_key")
DB_PATH = os.getenv("SHEETPULSE_DB_PATH", "sheetpulse.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=25.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            owner_name TEXT NOT NULL,
            tier TEXT DEFAULT 'free',
            credits_left INTEGER DEFAULT 100,
            total_used INTEGER DEFAULT 0,
            created_at REAL NOT NULL
        )
    """)
    cur.execute("""
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
    cur.execute("""
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
    cur.execute("SELECT key FROM api_keys WHERE key = 'sp_demo_live'")
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO api_keys VALUES ('sp_demo_live', 'Web Playground Sandbox', 'free', 10000, 0, ?)",
            (time.time(),)
        )
    conn.commit()
    conn.close()

init_db()

def is_byok_key(key: str) -> bool:
    k = key.strip()
    return k.startswith("gsk_") or k.startswith("csk-") or k.startswith("sk-") or k.startswith("AIza")

def verify_and_deduct_credits(api_key: str, amount: int = 1) -> Dict[str, Any]:
    sanitized_key = (api_key or "").strip()
    if not sanitized_key or sanitized_key == "YOUR_API_KEY_HERE":
        raise HTTPException(
            status_code=401,
            detail="Valid API Key required. Generate your key at https://sheetpulseai.onrender.com"
        )

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT owner_name, tier, credits_left, total_used FROM api_keys WHERE key = ?", (sanitized_key,))
        row = cur.fetchone()

        if not row and is_byok_key(sanitized_key):
            byok_provider = "Groq BYOK" if sanitized_key.startswith("gsk_") else "BYOK Provider"
            cur.execute(
                "INSERT INTO api_keys VALUES (?, ?, 'byok', 999999, ?, ?)",
                (sanitized_key, byok_provider, amount, time.time())
            )
            conn.commit()
            return {"owner": byok_provider, "tier": "BYOK", "remaining": 999999}

        if not row:
            raise HTTPException(status_code=401, detail="Invalid API Key. Key not recognized by cluster.")
        
        tier, credits_left = row["tier"], row["credits_left"]
        if tier not in ["developer", "byok"] and credits_left < amount:
            raise HTTPException(status_code=402, detail="Credit quota exhausted. Please top-up or upgrade your tier.")

        if tier == "byok":
            cur.execute("UPDATE api_keys SET total_used = total_used + ? WHERE key = ?", (amount, sanitized_key))
        else:
            cur.execute("UPDATE api_keys SET credits_left = credits_left - ?, total_used = total_used + ? WHERE key = ?", (amount, amount, sanitized_key))
        conn.commit()

        return {"owner": row["owner_name"], "tier": tier, "remaining": credits_left if tier == "byok" else credits_left - amount}
    finally:
        conn.close()

def log_request_event(api_key: str, action: str, provider: str, latency: float, input_len: int):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO usage_logs (key, action, provider, latency, input_length, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (api_key or "anonymous", action, provider, latency, input_len, time.time())
        )
        conn.commit()
        conn.close()
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
        req = urllib.request.Request(
            clean_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            text = re.sub(r'<script.*?</script>|<style.*?</style>|<header.*?</header>|<footer.*?</footer>|<nav.*?</nav>', '', html, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = ' '.join(text.split())
            return clean_text[:3500] if clean_text else "Empty content."
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
    api_key: Optional[str] = Field("")

class BatchRequest(BaseModel):
    items: List[ProcessRequest] = Field(..., max_length=100)
    api_key: Optional[str] = Field("")

def _sync_cerebras_call(sys_prompt: str, usr_prompt: str) -> Tuple[str, str]:
    if not CEREBRAS_API_KEY:
        raise ValueError("Cerebras unconfigured")
    url = "https://api.cerebras.ai/v1/chat/completions"
    payload = {
        "model": "llama3.1-8b",
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
        "temperature": 0.05,
        "max_tokens": 150
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {CEREBRAS_API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        res = json.loads(resp.read().decode())
        out = clean_output(res["choices"][0]["message"]["content"])
        if not out:
            raise ValueError("Empty output")
        return out, "Cerebras:Llama-3.1-8b"

def _sync_groq_call(sys_prompt: str, usr_prompt: str, custom_key: Optional[str] = None) -> Tuple[str, str]:
    effective_key = custom_key if (custom_key and custom_key.startswith("gsk_")) else GROQ_API_KEY
    if not effective_key:
        raise ValueError("Groq unconfigured")
    client = Groq(api_key=effective_key, timeout=8.0)
    for model_name in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
                temperature=0.05,
                max_tokens=150
            )
            out = clean_output(res.choices[0].message.content or "")
            if out:
                return out, f"Groq:{model_name}"
        except Exception:
            continue
    raise ValueError("Groq exhausted")

def _sync_openrouter_call(sys_prompt: str, usr_prompt: str) -> Tuple[str, str]:
    if not OPENROUTER_API_KEY:
        raise ValueError("OpenRouter unconfigured")
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
        "temperature": 0.05,
        "max_tokens": 150
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=9) as resp:
        res = json.loads(resp.read().decode())
        out = clean_output(res["choices"][0]["message"]["content"])
        if not out:
            raise ValueError("Empty output")
        return out, "OpenRouter:Llama-3.3-70b-Free"

def _sync_gemini_call(sys_prompt: str, usr_prompt: str) -> Tuple[str, str]:
    if not GEMINI_API_KEY:
        raise ValueError("Gemini unconfigured")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"{sys_prompt}\n\n{usr_prompt}"}]}],
        "generationConfig": {"temperature": 0.05, "maxOutputTokens": 150}
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=9) as response:
        res = json.loads(response.read().decode())
        out = clean_output(res["candidates"][0]["content"]["parts"][0]["text"])
        if not out:
            raise ValueError("Empty output")
        return out, "Google:Gemini-1.5-Flash"

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
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), SUM(total_used) FROM api_keys")
    u_count, total_exec = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM usage_logs")
    log_count = cur.fetchone()[0]
    conn.close()
    return {
        "status": "online",
        "service": "SheetPulse AI Enterprise Core",
        "version": "16.0.0",
        "active_keys": u_count or 0,
        "total_cells_processed": total_exec or 0,
        "logged_events": log_count or 0,
        "cache_entries": len(MEMORY_CACHE)
    }

# --- COMPLETE TELEMETRY DASHBOARD ---
@app.get("/api/v1/dashboard/stats")
def get_user_dashboard(api_key: str):
    sanitized_key = (api_key or "").strip()
    if not sanitized_key:
        raise HTTPException(status_code=400, detail="API Key parameter required")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT owner_name, tier, credits_left, total_used, created_at FROM api_keys WHERE key = ?", (sanitized_key,))
    key_row = cur.fetchone()

    if not key_row and is_byok_key(sanitized_key):
        byok_provider = "Groq BYOK" if sanitized_key.startswith("gsk_") else "BYOK Provider"
        cur.execute(
            "INSERT INTO api_keys VALUES (?, ?, 'byok', 999999, 0, ?)",
            (sanitized_key, byok_provider, time.time())
        )
        conn.commit()
        cur.execute("SELECT owner_name, tier, credits_left, total_used, created_at FROM api_keys WHERE key = ?", (sanitized_key,))
        key_row = cur.fetchone()

    if not key_row:
        conn.close()
        raise HTTPException(status_code=404, detail="API Key not found in cluster.")

    cur.execute(
        "SELECT action, provider, latency, timestamp FROM usage_logs WHERE key = ? ORDER BY id DESC LIMIT 15",
        (sanitized_key,)
    )
    recent_logs = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT COUNT(*), AVG(latency) FROM usage_logs WHERE key = ?", (sanitized_key,))
    log_count, avg_lat = cur.fetchone()
    avg_lat = avg_lat or 0.22

    # Accurate Total Cells Count from usage logs or db
    actual_total_used = max(key_row["total_used"], log_count or 0)

    conn.close()

    is_unlimited = key_row["tier"] == "byok"
    used_percentage = 0 if is_unlimited else min(100, round((actual_total_used / max(1, actual_total_used + key_row["credits_left"])) * 100))

    return {
        "success": True,
        "owner": key_row["owner_name"],
        "tier": "UNLIMITED BYOK" if is_unlimited else key_row["tier"].upper(),
        "credits_left": "Unlimited (BYOK)" if is_unlimited else key_row["credits_left"],
        "total_used": actual_total_used,
        "used_percentage": used_percentage,
        "avg_latency": f"{avg_lat:.2f}s",
        "created_at": time.strftime("%d %b %Y", time.localtime(key_row["created_at"])),
        "recent_logs": recent_logs
    }

@app.post("/api/v1/keys/new")
def create_free_api_key(req: KeyGenRequest):
    new_key = f"sp_{uuid.uuid4().hex[:18]}"
    initial_credits = 100 if req.tier == "free" else 5000
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO api_keys VALUES (?, ?, ?, ?, 0, ?)", 
        (new_key, req.owner_name.strip(), req.tier, initial_credits, time.time())
    )
    conn.commit()
    conn.close()
    return {
        "success": True,
        "api_key": new_key,
        "owner": req.owner_name,
        "credits": initial_credits,
        "tier": req.tier
    }

# --- PROCESS CELL PIPELINE ---
@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    start_time = time.time()
    text_content = (req.text or "").strip()
    if not text_content:
        return {"success": True, "result": "", "cached": False, "provider": "None"}

    # 1. Quota & Auth Check FIRST (Ensures counter increments for EVERY request)
    verify_and_deduct_credits(req.api_key or "", amount=1)

    # 2. Regex Fast Extractor
    if req.action == "extract" and req.instruction:
        fast_match = try_fast_regex_extract(text_content, req.instruction)
        if fast_match:
            elapsed = round(time.time() - start_time, 3)
            log_request_event(req.api_key or "anonymous", "extract", "Regex:UltraFast", elapsed, len(text_content))
            return {"success": True, "result": fast_match, "provider": "Regex:UltraFast", "cached": False}

    # 3. In-Memory Cache
    cache_key = hashlib.sha256(f"{req.action}:{req.instruction}:{text_content}".lower().encode()).hexdigest()
    if cache_key in MEMORY_CACHE:
        elapsed = round(time.time() - start_time, 3)
        log_request_event(req.api_key or "anonymous", req.action, "Memory:Cache", elapsed, len(text_content))
        return {
            "success": True,
            "result": MEMORY_CACHE[cache_key]["result"],
            "provider": MEMORY_CACHE[cache_key]["provider"],
            "cached": True
        }

    sys_prompt, usr_prompt = resolve_action_prompts(req.action, req.instruction or "", text_content)

    async with CONCURRENCY_SEMAPHORE:
        result, provider = None, None
        
        if req.api_key and req.api_key.startswith("gsk_"):
            try:
                result, provider = await asyncio.to_thread(_sync_groq_call, sys_prompt, usr_prompt, req.api_key)
            except Exception as ge:
                raise HTTPException(status_code=400, detail=f"Groq API Error: {str(ge)}")
        else:
            try:
                result, provider = await asyncio.to_thread(_sync_cerebras_call, sys_prompt, usr_prompt)
            except Exception:
                try:
                    result, provider = await asyncio.to_thread(_sync_groq_call, sys_prompt, usr_prompt)
                except Exception:
                    try:
                        result, provider = await asyncio.to_thread(_sync_openrouter_call, sys_prompt, usr_prompt)
                    except Exception:
                        try:
                            result, provider = await asyncio.to_thread(_sync_gemini_call, sys_prompt, usr_prompt)
                        except Exception as e:
                            raise HTTPException(status_code=503, detail=f"Inference cluster busy: {str(e)}")

        if len(MEMORY_CACHE) >= MAX_CACHE_ENTRIES:
            MEMORY_CACHE.pop(next(iter(MEMORY_CACHE)))
        MEMORY_CACHE[cache_key] = {"result": result, "provider": provider}

        elapsed = round(time.time() - start_time, 3)
        log_request_event(req.api_key or "anonymous", req.action, provider, elapsed, len(text_content))

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
