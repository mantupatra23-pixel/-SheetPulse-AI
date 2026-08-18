import os
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

# Fallback models priority list
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768"
]

class ProcessRequest(BaseModel):
    text: str
    instruction: str
    action: str = "custom"

@app.get("/")
def health_check():
    return {"status": "online", "service": "SheetPulse AI Backend", "version": "1.3.0"}

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": ""}

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    if req.action == "extract":
        system_prompt = "You are a precise data extractor. Output ONLY the extracted text with no extra words."
        user_prompt = f"Target: {req.instruction}\nInput: {req.text}"
    elif req.action == "clean":
        system_prompt = "You are a data cleaner. Clean and standardize the input text. Output ONLY the cleaned text."
        user_prompt = f"Input: {req.text}"
    elif req.action == "classify":
        system_prompt = f"Classify input into: [{req.instruction}]. Output ONLY the single exact matched label."
        user_prompt = f"Input: {req.text}"
    else:
        system_prompt = "You are SheetPulse AI. Follow instruction precisely and output only the direct result."
        user_prompt = f"Instruction: {req.instruction}\nContext: {req.text}"

    client = Groq(api_key=GROQ_API_KEY)
    
    last_error = ""
    for model_name in FALLBACK_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=250
            )
            return {
                "success": True,
                "result": completion.choices[0].message.content.strip(),
                "model_used": model_name
            }
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=500, detail=f"All models failed. Last error: {last_error}")
