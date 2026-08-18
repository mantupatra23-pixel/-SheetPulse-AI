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

class ProcessRequest(BaseModel):
    text: str
    instruction: str
    action: str = "custom"

@app.get("/")
def health_check():
    return {"status": "online", "service": "SheetPulse AI Backend", "version": "1.4.0"}

@app.get("/api/v1/models")
def get_available_models():
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY missing"}
    try:
        client = Groq(api_key=GROQ_API_KEY)
        models = [m.id for m in client.models.list().data if "whisper" not in m.id.lower()]
        return {"available_models": models}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": ""}

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    if req.action == "extract":
        system_prompt = "You are a precise data extractor. Output ONLY the extracted text with no extra words or markdown."
        user_prompt = f"Target: {req.instruction}\nInput: {req.text}"
    elif req.action == "clean":
        system_prompt = "You are a data cleaner. Clean and standardize the input text. Output ONLY the cleaned text."
        user_prompt = f"Input: {req.text}"
    elif req.action == "classify":
        system_prompt = f"Classify input into: [{req.instruction}]. Output ONLY the exact matched tag."
        user_prompt = f"Input: {req.text}"
    else:
        system_prompt = "You are SheetPulse AI. Follow instruction precisely and output only the direct result."
        user_prompt = f"Instruction: {req.instruction}\nContext: {req.text}"

    client = Groq(api_key=GROQ_API_KEY)

    try:
        # Dynamically fetch current active chat models from Groq account
        model_list = [m.id for m in client.models.list().data if "whisper" not in m.id.lower()]
    except Exception:
        model_list = ["qwen-2.5-32b", "gemma2-9b-it"]

    last_error = ""
    for model_name in model_list:
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
