import os
import re
import time
import hashlib
import asyncio
import sqlite3
import urllib.request
import json
import uuid
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="SheetPulse AI Enterprise Core", version="9.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DB_PATH = "sheetpulse.db"

# --- Database & Persistent Storage ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            owner_name TEXT,
            tier TEXT DEFAULT 'free',
            credits_left INTEGER DEFAULT 100,
            total_used INTEGER DEFAULT 0,
            created_at REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            action TEXT,
            provider TEXT,
            latency REAL,
            timestamp REAL
        )
    """)
    cur.execute("SELECT key FROM api_keys WHERE key = 'sp_demo_live'")
    if not cur.fetchone():
        cur.execute("INSERT INTO api_keys VALUES ('sp_demo_live', 'Developer Demo', 'developer', 100000, 0, ?)", (time.time(),))
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def verify_and_deduct_credits(api_key: str, amount: int = 1):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT credits_left, tier FROM api_keys WHERE key = ?", (api_key,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid API Key. Generate one at the landing dashboard.")
    
    credits_left, tier = row["credits_left"], row["tier"]
    if tier != "developer" and credits_left < amount:
        conn.close()
        raise HTTPException(status_code=402, detail="Credit quota exhausted. Top-up or upgrade plan.")

    cur.execute("UPDATE api_keys SET credits_left = credits_left - ?, total_used = total_used + ? WHERE key = ?", (amount, amount, api_key))
    conn.commit()
    conn.close()

def log_request(api_key: str, action: str, provider: str, latency: float):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO usage_logs (key, action, provider, latency, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (api_key, action, provider, latency, time.time()))
        conn.commit()
        conn.close()
    except Exception:
        pass

# --- High-Performance Memory Cache & Concurrency Limiter ---
MEMORY_CACHE: Dict[str, dict] = {}
MAX_CACHE_ENTRIES = 3000
CONCURRENCY_LIMIT = asyncio.Semaphore(30)

# --- Zero-Token Regex Smart Extractor ---
REGEX_PATTERNS = {
    "email": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    "phone": r'(\+?[0-9]{1,3}[-.\s]?)?\(?[0-9]{2,5}\)?[-.\s]?[0-9]{3,5}[-.\s]?[0-9]{3,5}',
    "url": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
    "price": r'[\$\€\£\₹]\s?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?',
    "date": r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
    "number": r'\b\d+(?:\.\d+)?\b'
}

def try_fast_regex_extract(text: str, target: str) -> Optional[str]:
    target_lower = target.lower().strip()
    for key, pattern in REGEX_PATTERNS.items():
        if key in target_lower:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
    return None

def fetch_url_text(url: str) -> str:
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=7) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            text = re.sub(r'<script.*?</script>|<style.*?</style>|<header.*?</header>|<footer.*?</footer>', '', html, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            return ' '.join(text.split())[:3000]
    except Exception as e:
        return f"Scrape Error: {str(e)}"

def clean_output(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    return text.strip().strip('"\'`').strip()

# --- Request / Response Models ---
class KeyGenRequest(BaseModel):
    owner_name: str
    tier: Optional[str] = "free"

class KeyRefillRequest(BaseModel):
    api_key: str
    credits_to_add: int

class ProcessRequest(BaseModel):
    text: str
    instruction: Optional[str] = ""
    action: str = "custom"
    api_key: Optional[str] = "sp_demo_live"

class BatchRequest(BaseModel):
    items: List[ProcessRequest]
    api_key: Optional[str] = "sp_demo_live"

# --- Provider Execution Engines ---
def call_cerebras_engine(sys_prompt: str, usr_prompt: str):
    if not CEREBRAS_API_KEY:
        raise Exception("CEREBRAS_API_KEY missing")
    url = "https://api.cerebras.ai/v1/chat/completions"
    payload = {
        "model": "llama3.1-8b",
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
        "temperature": 0.1,
        "max_tokens": 250
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {CEREBRAS_API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        res = json.loads(resp.read().decode())
        return clean_output(res["choices"][0]["message"]["content"]), "Cerebras:Llama-3.1-8b"

def call_groq_engine(sys_prompt: str, usr_prompt: str):
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY missing")
    client = Groq(api_key=GROQ_API_KEY)
    all_models = client.models.list().data
    active_models = [m.id for m in all_models if not any(x in m.id.lower() for x in ["whisper", "guard", "vision", "embed"])]
    for model_name in active_models:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
                temperature=0.1,
                max_tokens=250
            )
            raw = res.choices[0].message.content or ""
            cleaned = clean_output(raw)
            if cleaned:
                return cleaned, f"Groq:{model_name}"
        except Exception:
            continue
    raise Exception("Groq busy")

def call_openrouter_engine(sys_prompt: str, usr_prompt: str):
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY missing")
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
        "temperature": 0.1,
        "max_tokens": 250
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://sheetpulseai.onrender.com",
            "X-Title": "SheetPulse AI"
        }
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode())
        return clean_output(res["choices"][0]["message"]["content"]), "OpenRouter:Llama-3.3-70b-Free"

def call_gemini_engine(sys_prompt: str, usr_prompt: str):
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"{sys_prompt}\n\n{usr_prompt}"}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 250}
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        res = json.loads(response.read().decode())
        raw = res["candidates"][0]["content"]["parts"][0]["text"]
        return clean_output(raw), "Google:Gemini-1.5-Flash"

# --- SYSTEM PROMPT BUILDER ---
def resolve_action_prompts(action: str, instruction: str, text: str):
    action = action.lower()
    if action == "clean":
        sys = "You are a precise data standardizer. Standardize casing, fix spacing, format emails and clean data. Output ONLY the cleaned result."
        usr = f"Input: {text}"
    elif action == "extract":
        sys = "You are an entity extractor. Output ONLY the extracted text with no conversational labels."
        usr = f"Target: {instruction}\nText: {text}"
    elif action == "classify":
        sys = f"Classify input strictly into ONE tag from: [{instruction}]. Output ONLY the exact tag name."
        usr = f"Input: {text}"
    elif action == "translate":
        sys = f"Translate input accurately to target language: {instruction}. Output ONLY the direct translated text."
        usr = f"Input: {text}"
    elif action == "summarize":
        sys = "Summarize the text concisely. Output ONLY the concise summary."
        usr = f"Instruction: {instruction}\nText: {text}"
    elif action == "fix_formula":
        sys = "Analyze the broken spreadsheet formula and fix it. Output ONLY a valid working formula starting with '='."
        usr = f"Broken Formula: {text}\nContext/Goal: {instruction}"
    elif action == "formula":
        sys = "Generate a valid Google Sheets formula starting with '=' based on description. Output ONLY the formula."
        usr = f"Requirement: {instruction}\nColumns/Context: {text}"
    elif action == "list":
        sys = "Generate a comma-separated list of items based on instructions. Output ONLY items separated by comma."
        usr = f"Topic/Context: {text}\nInstruction: {instruction}"
    elif action == "scrape":
        scraped_data = fetch_url_text(text.strip())
        sys = "Analyze web page content and answer the target question directly. Output ONLY the direct answer."
        usr = f"Question: {instruction}\nWeb Page Content:\n{scraped_data}"
    else:
        sys = "You are SheetPulse AI. Execute the instruction directly and output ONLY the direct final answer."
        usr = f"Instruction: {instruction}\nContext: {text}"
    return sys, usr

# --- ROUTE 1: FRONTEND SERVING ---
@app.get("/")
def serve_home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>SheetPulse AI Backend Online</h1>")

# --- ROUTE 2: HEALTH & TELEMETRY ---
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
        "version": "9.0.0",
        "active_keys": u_count or 0,
        "total_cells_processed": total_exec or 0,
        "logged_events": log_count or 0,
        "cache_entries": len(MEMORY_CACHE),
        "cluster_health": {
            "cerebras": bool(CEREBRAS_API_KEY),
            "groq": bool(GROQ_API_KEY),
            "openrouter": bool(OPENROUTER_API_KEY),
            "gemini": bool(GEMINI_API_KEY)
        }
    }

# --- ROUTE 3: API KEY GENERATION & BILLING ---
@app.post("/api/v1/keys/new")
def create_api_key(req: KeyGenRequest):
    new_key = f"sp_{uuid.uuid4().hex[:18]}"
    initial_credits = 100 if req.tier == "free" else 5000
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO api_keys VALUES (?, ?, ?, ?, 0, ?)", (new_key, req.owner_name, req.tier, initial_credits, time.time()))
    conn.commit()
    conn.close()
    return {"success": True, "api_key": new_key, "owner": req.owner_name, "credits": initial_credits, "tier": req.tier}

@app.get("/api/v1/keys/balance")
def check_balance(api_key: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT owner_name, tier, credits_left, total_used, created_at FROM api_keys WHERE key = ?", (api_key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="API Key not found")
    return dict(row)

@app.post("/api/v1/keys/refill")
def refill_credits(req: KeyRefillRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT credits_left FROM api_keys WHERE key = ?", (req.api_key,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="API Key not found")
    cur.execute("UPDATE api_keys SET credits_left = credits_left + ? WHERE key = ?", (req.credits_to_add, req.api_key))
    conn.commit()
    conn.close()
    return {"success": True, "api_key": req.api_key, "added": req.credits_to_add}

# --- ROUTE 4: SINGLE CELL CORE INFERENCE ---
@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    start_time = time.time()
    if not req.text or not req.text.strip():
        return {"success": True, "result": "", "cached": False, "provider": "None"}

    # 1. Regex Fast Extractor Bypass
    if req.action == "extract" and req.instruction:
        fast_match = try_fast_regex_extract(req.text, req.instruction)
        if fast_match:
            elapsed = round(time.time() - start_time, 3)
            log_request(req.api_key or "sp_demo_live", "extract", "Regex:UltraFast", elapsed)
            return {"success": True, "result": fast_match, "provider": "Regex:UltraFast", "cached": False}

    # 2. In-Memory Cache Lookup
    cache_key = hashlib.sha256(f"{req.action}:{req.instruction}:{req.text}".lower().encode()).hexdigest()
    if cache_key in MEMORY_CACHE:
        elapsed = round(time.time() - start_time, 3)
        log_request(req.api_key or "sp_demo_live", req.action, "Memory:Cache", elapsed)
        return {"success": True, "result": MEMORY_CACHE[cache_key]["result"], "provider": MEMORY_CACHE[cache_key]["provider"], "cached": True}

    # 3. Credit Verification & Deduction
    verify_and_deduct_credits(req.api_key or "sp_demo_live", amount=1)

    sys_prompt, usr_prompt = resolve_action_prompts(req.action, req.instruction or "", req.text)

    async with CONCURRENCY_LIMIT:
        result, provider = None, None
        # Cascade Priority: Cerebras -> Groq -> OpenRouter -> Gemini
        try:
            result, provider = call_cerebras_engine(sys_prompt, usr_prompt)
        except Exception:
            try:
                result, provider = call_groq_engine(sys_prompt, usr_prompt)
            except Exception:
                try:
                    result, provider = call_openrouter_engine(sys_prompt, usr_prompt)
                except Exception:
                    try:
                        result, provider = call_gemini_engine(sys_prompt, usr_prompt)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"All multi-cluster providers failed: {e}")

        # Cache Store
        if len(MEMORY_CACHE) >= MAX_CACHE_ENTRIES:
            MEMORY_CACHE.pop(next(iter(MEMORY_CACHE)))
        MEMORY_CACHE[cache_key] = {"result": result, "provider": provider}

        elapsed = round(time.time() - start_time, 3)
        log_request(req.api_key or "sp_demo_live", req.action, provider, elapsed)

        return {"success": True, "result": result, "provider": provider, "cached": False}

# --- ROUTE 5: PARALLEL BATCH PROCESSOR ---
@app.post("/api/v1/batch")
async def process_batch(batch: BatchRequest):
    async def worker(item: ProcessRequest):
        try:
            item.api_key = batch.api_key
            return await process_cell(item)
        except Exception as e:
            return {"success": False, "error": str(e)}

    results = await asyncio.gather(*[worker(item) for item in batch.items])
    return {"success": True, "processed_count": len(results), "data": results}
