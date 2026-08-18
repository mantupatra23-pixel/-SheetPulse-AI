import os
import re
import hashlib
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="SheetPulse AI Enterprise Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# In-memory fast cache (Key: SHA256 -> Value: Result)
MEMORY_CACHE = {}
MAX_CACHE_SIZE = 1000

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

class BatchRequest(BaseModel):
    items: List[ProcessRequest]

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "SheetPulse AI Enterprise",
        "version": "2.0.0",
        "cached_queries": len(MEMORY_CACHE)
    }

def execute_groq_inference(client: Groq, system_prompt: str, user_prompt: str):
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
                max_tokens=350
            )
            raw = res.choices[0].message.content or ""
            cleaned = clean_output(raw)
            if cleaned:
                return cleaned, model_name
        except Exception as e:
            last_err = str(e)
            continue
    raise Exception(f"All inference models failed. Details: {last_err}")

def resolve_prompts(action: str, instruction: str, text: str):
    action = action.lower()
    if action == "clean":
        sys = "You are an automated data standardizer. Clean casing, remove broken spaces, standardize phone/email/addresses. Output ONLY cleaned data."
        usr = f"Input: {text}"
    elif action == "extract":
        sys = "You are a precise entity extractor. Output ONLY the exact extracted entity without formatting or extra labels."
        usr = f"Target Entity: {instruction}\nSource Content: {text}"
    elif action == "classify":
        sys = f"Classify the text strictly into ONE tag from: [{instruction}]. Output ONLY the exact tag string."
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
        sys = "You are SheetPulse AI. Execute the instruction directly and output ONLY the direct final answer."
        usr = f"Instruction: {instruction}\nContext: {text}"
    return sys, usr

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": "", "cached": False}

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable missing")

    cache_key = get_cache_key(req.text, req.instruction or "", req.action)
    if cache_key in MEMORY_CACHE:
        return {
            "success": True,
            "result": MEMORY_CACHE[cache_key]["result"],
            "model_used": MEMORY_CACHE[cache_key]["model"],
            "cached": True
        }

    sys_prompt, usr_prompt = resolve_prompts(req.action, req.instruction or "", req.text)
    client = Groq(api_key=GROQ_API_KEY)

    try:
        result, model_name = execute_groq_inference(client, sys_prompt, usr_prompt)
        
        # Cache write
        if len(MEMORY_CACHE) >= MAX_CACHE_SIZE:
            MEMORY_CACHE.pop(next(iter(MEMORY_CACHE)))
        MEMORY_CACHE[cache_key] = {"result": result, "model": model_name}

        return {"success": True, "result": result, "model_used": model_name, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch")
async def process_batch(batch: BatchRequest):
    async def worker(item: ProcessRequest):
        try:
            return await process_cell(item)
        except Exception as e:
            return {"success": False, "error": str(e)}

    results = await asyncio.gather(*[worker(item) for item in batch.items])
    return {"success": True, "total": len(results), "data": results}
