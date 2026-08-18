import os
import re
import time
import hashlib
import asyncio
import urllib.request
import json
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="SheetPulse AI Enterprise Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# In-Memory Cache & Analytics
MEMORY_CACHE: Dict[str, dict] = {}
MAX_CACHE_SIZE = 2000
API_USAGE_STATS = {
    "total_requests": 0,
    "cache_hits": 0,
    "groq_executions": 0,
    "gemini_failovers": 0,
    "errors": 0
}

# Concurrency Semaphore to prevent rate spikes
CONCURRENCY_LIMIT = asyncio.Semaphore(15)

def get_cache_key(text: str, instruction: str, action: str) -> str:
    raw = f"{action}:{instruction}:{text}".strip().lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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

class ProcessRequest(BaseModel):
    text: str
    instruction: Optional[str] = ""
    action: str = "custom"
    api_key: Optional[str] = "demo_key"

class BatchRequest(BaseModel):
    items: List[ProcessRequest]
    api_key: Optional[str] = "demo_key"

def resolve_prompts(action: str, instruction: str, text: str):
    action = action.lower()
    if action == "clean":
        sys = "You are a data cleaner. Standardize formatting, fix broken casing/spaces, clean emails/phones. Output ONLY cleaned data."
        usr = f"Input: {text}"
    elif action == "extract":
        sys = "You are an entity extractor. Output ONLY the exact extracted entity without formatting or conversational filler."
        usr = f"Target Entity: {instruction}\nSource Content: {text}"
    elif action == "classify":
        sys = f"Classify the text strictly into ONE tag from: [{instruction}]. Output ONLY the exact single tag string."
        usr = f"Input: {text}"
    elif action == "translate":
        sys = f"Translate input accurately to target language: {instruction}. Output ONLY the translated text."
        usr = f"Source Text: {text}"
    elif action == "summarize":
        sys = "Summarize the text concisely. Output ONLY the concise summary."
        usr = f"Instruction: {instruction}\nText: {text}"
    elif action == "formula":
        sys = "You are a Google Sheets formula generator. Output ONLY the working formula starting with '='."
        usr = f"Task Requirement: {instruction}\nContext/Columns: {text}"
    else:
        sys = "You are SheetPulse AI. Execute the instruction directly and output ONLY the final direct answer."
        usr = f"Instruction: {instruction}\nContext: {text}"
    return sys, usr

def call_groq_engine(system_prompt: str, user_prompt: str):
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not configured")
    
    client = Groq(api_key=GROQ_API_KEY)
    try:
        all_models = client.models.list().data
        active_models = [
            m.id for m in all_models 
            if not any(x in m.id.lower() for x in ["whisper", "guard", "vision", "embed"])
        ]
    except Exception:
        active_models = ["llama-3.3-70b-versatile", "gemma2-9b-it"]

    last_err = ""
    for model_name in active_models:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            raw = res.choices[0].message.content or ""
            cleaned = clean_output(raw)
            if cleaned:
                return cleaned, f"Groq:{model_name}"
        except Exception as e:
            last_err = str(e)
            continue
    raise Exception(f"Groq cluster failed: {last_err}")

def call_gemini_fallback(system_prompt: str, user_prompt: str):
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not configured")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300}
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        res = json.loads(response.read().decode())
        raw = res["candidates"][0]["content"]["parts"][0]["text"]
        return clean_output(raw), "Google:Gemini-1.5-Flash"

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "SheetPulse AI Enterprise",
        "version": "3.0.0",
        "cache_entries": len(MEMORY_CACHE),
        "analytics": API_USAGE_STATS
    }

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    API_USAGE_STATS["total_requests"] += 1

    if not req.text or not req.text.strip():
        return {"success": True, "result": "", "cached": False}

    cache_key = get_cache_key(req.text, req.instruction or "", req.action)
    if cache_key in MEMORY_CACHE:
        API_USAGE_STATS["cache_hits"] += 1
        return {
            "success": True,
            "result": MEMORY_CACHE[cache_key]["result"],
            "provider": MEMORY_CACHE[cache_key]["provider"],
            "cached": True
        }

    sys_prompt, usr_prompt = resolve_prompts(req.action, req.instruction or "", req.text)

    async with CONCURRENCY_LIMIT:
        result, provider = None, None
        # Tier 1: Groq Engine
        try:
            result, provider = call_groq_engine(sys_prompt, usr_prompt)
            API_USAGE_STATS["groq_executions"] += 1
        except Exception as groq_err:
            # Tier 2: Gemini Failover
            try:
                result, provider = call_gemini_fallback(sys_prompt, usr_prompt)
                API_USAGE_STATS["gemini_failovers"] += 1
            except Exception as gem_err:
                API_USAGE_STATS["errors"] += 1
                raise HTTPException(status_code=500, detail=f"All providers exhausted. Groq: {groq_err} | Gemini: {gem_err}")

        # Cache write (LRU style drop)
        if len(MEMORY_CACHE) >= MAX_CACHE_SIZE:
            MEMORY_CACHE.pop(next(iter(MEMORY_CACHE)))
        MEMORY_CACHE[cache_key] = {"result": result, "provider": provider}

        return {
            "success": True,
            "result": result,
            "provider": provider,
            "cached": False
        }

@app.post("/api/v1/batch")
async def process_batch(batch: BatchRequest):
    async def worker(item: ProcessRequest):
        try:
            return await process_cell(item)
        except Exception as e:
            return {"success": False, "error": str(e)}

    results = await asyncio.gather(*[worker(item) for item in batch.items])
    return {
        "success": True,
        "processed_count": len(results),
        "data": results
    }
