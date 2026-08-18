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

def clean_output(text: str) -> str:
    if not text:
        return ""
    # Agar thinking tags hain to sirf final answer extract karo
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    
    # Extra quotes, backticks aur whitespace strip karo
    text = text.strip().strip('"\'`')
    return text.strip()

class ProcessRequest(BaseModel):
    text: str
    instruction: str
    action: str = "custom"

@app.get("/")
def health_check():
    return {"status": "online", "service": "SheetPulse AI Backend", "version": "1.7.0"}

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": ""}

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing in Render environment")

    if req.action == "extract":
        system_prompt = "You are a precise data extractor. Output ONLY the extracted text. Zero extra words or thinking."
        user_prompt = f"Target to extract: {req.instruction}\nInput data: {req.text}"
    elif req.action == "clean":
        system_prompt = "You are a data cleaner. Clean spaces, casing, and format standard values. Output ONLY the cleaned result."
        user_prompt = f"Input: {req.text}"
    elif req.action == "classify":
        system_prompt = f"Classify input into exactly one of: [{req.instruction}]. Output ONLY the exact tag."
        user_prompt = f"Input: {req.text}"
    else:
        system_prompt = "You are SheetPulse AI. Execute the instruction directly and output ONLY the answer."
        user_prompt = f"Instruction: {req.instruction}\nContext: {req.text}"

    client = Groq(api_key=GROQ_API_KEY)

    try:
        # Dynamically get active chat models
        all_models = client.models.list().data
        active_models = [
            m.id for m in all_models 
            if not any(x in m.id.lower() for x in ["whisper", "guard", "vision", "embed"])
        ]
    except Exception as e:
        active_models = ["llama-3.3-70b-versatile", "gemma2-9b-it"]

    last_err = ""
    for model_name in active_models:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            raw_text = completion.choices[0].message.content or ""
            cleaned = clean_output(raw_text)
            if cleaned:
                return {
                    "success": True,
                    "result": cleaned,
                    "model_used": model_name
                }
        except Exception as err:
            last_err = str(err)
            continue

    raise HTTPException(status_code=500, detail=f"Inference failed. Error: {last_err}")
