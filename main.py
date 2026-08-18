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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="SheetPulse AI Triple-Engine SaaS", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DB_PATH = "sheetpulse.db"

# --- SQLite Database ---
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
    cur.execute("SELECT key FROM api_keys WHERE key = 'sp_demo_live'")
    if not cur.fetchone():
        cur.execute("INSERT INTO api_keys VALUES ('sp_demo_live', 'Developer Demo', 'developer', 10000, 0, ?)", (time.time(),))
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
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    credits_left, tier = row["credits_left"], row["tier"]
    if tier != "developer" and credits_left < amount:
        conn.close()
        raise HTTPException(status_code=402, detail="Credits exhausted")

    cur.execute("UPDATE api_keys SET credits_left = credits_left - ?, total_used = total_used + ? WHERE key = ?", (amount, amount, api_key))
    conn.commit()
    conn.close()

MEMORY_CACHE: Dict[str, dict] = {}
CONCURRENCY_LIMIT = asyncio.Semaphore(25)

REGEX_PATTERNS = {
    "email": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    "phone": r'(\+?[0-9]{1,3}[-.\s]?)?\(?[0-9]{2,5}\)?[-.\s]?[0-9]{3,5}[-.\s]?[0-9]{3,5}',
    "url": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
    "price": r'[\$\€\£\₹]\s?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?'
}

def try_fast_regex_extract(text: str, target: str) -> Optional[str]:
    target_lower = target.lower().strip()
    for key, pattern in REGEX_PATTERNS.items():
        if key in target_lower:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
    return None

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

class KeyGenRequest(BaseModel):
    owner_name: str
    tier: Optional[str] = "free"

class ProcessRequest(BaseModel):
    text: str
    instruction: Optional[str] = ""
    action: str = "custom"
    api_key: Optional[str] = "sp_demo_live"

class BatchRequest(BaseModel):
    items: List[ProcessRequest]
    api_key: Optional[str] = "sp_demo_live"

# --- PROVIDER 1: Cerebras (Ultra-Fast 1800 Tok/s) ---
def call_cerebras_engine(sys_prompt: str, usr_prompt: str):
    if not CEREBRAS_API_KEY:
        raise Exception("CEREBRAS_API_KEY missing")
    
    url = "https://api.cerebras.ai/v1/chat/completions"
    payload = {
        "model": "llama3.1-8b",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 250
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CEREBRAS_API_KEY}"
        }
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode())
        raw = res["choices"][0]["message"]["content"]
        return clean_output(raw), "Cerebras:Llama-3.1-8b"

# --- PROVIDER 2: Groq Engine ---
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
    raise Exception("Groq cluster busy")

# --- PROVIDER 3: OpenRouter (Universal Free Pool) ---
def call_openrouter_engine(sys_prompt: str, usr_prompt: str):
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY missing")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt}
        ],
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
    with urllib.request.urlopen(req, timeout=12) as resp:
        res = json.loads(resp.read().decode())
        raw = res["choices"][0]["message"]["content"]
        return clean_output(raw), "OpenRouter:Llama-3.3-70b-Free"

@app.get("/")
def system_status():
    return {
        "status": "online",
        "service": "SheetPulse AI Triple Engine",
        "version": "5.0.0",
        "providers_configured": {
            "cerebras": bool(CEREBRAS_API_KEY),
            "groq": bool(GROQ_API_KEY),
            "openrouter": bool(OPENROUTER_API_KEY)
        },
        "cache_entries": len(MEMORY_CACHE)
    }

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
    cur.execute("SELECT owner_name, tier, credits_left, total_used FROM api_keys WHERE key = ?", (api_key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="API Key not found")
    return dict(row)

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": "", "cached": False, "provider": "None"}

    if req.action == "extract" and req.instruction:
        fast_match = try_fast_regex_extract(req.text, req.instruction)
        if fast_match:
            return {"success": True, "result": fast_match, "provider": "Regex:UltraFast", "cached": False}

    cache_key = hashlib.sha256(f"{req.action}:{req.instruction}:{req.text}".lower().encode()).hexdigest()
    if cache_key in MEMORY_CACHE:
        return {"success": True, "result": MEMORY_CACHE[cache_key]["result"], "provider": MEMORY_CACHE[cache_key]["provider"], "cached": True}

    verify_and_deduct_credits(req.api_key or "sp_demo_live", amount=1)

    act = req.action.lower()
    if act == "clean":
        sys = "Standardize formatting, fix broken spacing/casing, clean text. Output ONLY cleaned result."
        usr = f"Input: {req.text}"
    elif act == "extract":
        sys = "Extract the exact requested entity. Output ONLY the extracted text."
        usr = f"Target: {req.instruction}\nText: {req.text}"
    elif act == "classify":
        sys = f"Classify input strictly into ONE tag from: [{req.instruction}]. Output ONLY the exact tag name."
        usr = f"Input: {req.text}"
    elif act == "translate":
        sys = f"Translate input accurately to: {req.instruction}. Output ONLY the translation."
        usr = f"Input: {req.text}"
    elif act == "formula":
        sys = "Generate a valid Google Sheets formula starting with '='. Output ONLY the formula."
        usr = f"Requirement: {req.instruction}\nContext: {req.text}"
    else:
        sys = "Execute the instruction directly. Output ONLY the final answer."
        usr = f"Instruction: {req.instruction}\nContext: {req.text}"

    async with CONCURRENCY_LIMIT:
        result, provider = None, None
        # Tier 1: Cerebras
        try:
            result, provider = call_cerebras_engine(sys, usr)
        except Exception:
            # Tier 2: Groq
            try:
                result, provider = call_groq_engine(sys, usr)
            except Exception:
                # Tier 3: OpenRouter
                try:
                    result, provider = call_openrouter_engine(sys, usr)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"All 3 AI clusters exhausted: {e}")

        if len(MEMORY_CACHE) >= 2000:
            MEMORY_CACHE.pop(next(iter(MEMORY_CACHE)))
        MEMORY_CACHE[cache_key] = {"result": result, "provider": provider}

        return {"success": True, "result": result, "provider": provider, "cached": False}

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
