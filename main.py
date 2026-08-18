import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="SheetPulse AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Pure fast chat models list (No reasoning delay / No <think> tags)
FAST_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "qwen-2.5-32b"
]

def clean_ai_output(raw_text: str) -> str:
    # 1. Strip closed <think> tags
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    # 2. Strip unclosed <think> tags if model truncated
    if "<think>" in cleaned:
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    return cleaned.strip().strip('"\'`').strip()

class ProcessRequest(BaseModel):
    text: str
    instruction: str
    action: str = "custom"

@app.get("/")
def health_check():
    return {"status": "online", "service": "SheetPulse AI Backend", "version": "1.6.0"}

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": ""}

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    if req.action == "extract":
        system_prompt = "You are an extractor. Output ONLY the extracted text. Zero explanations."
        user_prompt = f"Target: {req.instruction}\nInput: {req.text}"
    elif req.action == "clean":
        system_prompt = "You are a data cleaner. Standardize formatting and fix spaces/casing. Output ONLY the cleaned string."
        user_prompt = f"Input: {req.text}"
    elif req.action == "classify":
        system_prompt = f"Classify input into exact one tag: [{req.instruction}]. Output ONLY the exact tag name."
        user_prompt = f"Input: {req.text}"
    else:
        system_prompt = "You are SheetPulse AI. Output only the direct answer to instruction."
        user_prompt = f"Instruction: {req.instruction}\nContext: {req.text}"

    client = Groq(api_key=GROQ_API_KEY)

    last_error = ""
    for model_name in FAST_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            raw_result = completion.choices[0].message.content or ""
            clean_result = clean_ai_output(raw_result)
            if clean_result:
                return {
                    "success": True,
                    "result": clean_result,
                    "model_used": model_name
                }
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=500, detail=f"All models failed. Last error: {last_error}")
