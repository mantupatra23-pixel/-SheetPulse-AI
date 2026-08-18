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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

class ProcessRequest(BaseModel):
    text: str
    instruction: str
    action: str = "custom"  # custom, extract, clean, classify

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "SheetPulse AI Backend",
        "active_model": GROQ_MODEL,
        "version": "1.2.0"
    }

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": ""}

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server")

    if req.action == "extract":
        system_prompt = "You are a precise data extractor. Extract only the requested entity from the input. Output ONLY the extracted text with no explanations, no formatting, and no introductory remarks."
        user_prompt = f"Target to extract: {req.instruction}\nInput text: {req.text}"
    elif req.action == "clean":
        system_prompt = "You are a data cleaner. Clean, standardize, and format the input text. Fix capitalization, spaces, typos, and formatting. Output ONLY the cleaned text with no extra comments."
        user_prompt = f"Input text: {req.text}"
    elif req.action == "classify":
        system_prompt = f"You are a strict data classifier. Classify the input text into one of these categories: [{req.instruction}]. Output ONLY the category name exactly as matched."
        user_prompt = f"Input text: {req.text}"
    else:
        system_prompt = "You are SheetPulse AI, an ultra-fast spreadsheet AI assistant. Follow the user instruction precisely and output only the direct result with zero conversational filler."
        user_prompt = f"Instruction: {req.instruction}\nContext/Input: {req.text}"

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        result_text = completion.choices[0].message.content.strip()
        return {"success": True, "result": result_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
