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

def clean_ai_output(raw_text: str) -> str:
    # Remove <think>...</think> reasoning blocks
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    # Strip quotes, backticks and excessive blank lines
    cleaned = cleaned.strip().strip('"').strip("'").strip('`')
    return cleaned.strip()

class ProcessRequest(BaseModel):
    text: str
    instruction: str
    action: str = "custom"

@app.get("/")
def health_check():
    return {"status": "online", "service": "SheetPulse AI Backend", "version": "1.5.0"}

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": ""}

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    if req.action == "extract":
        system_prompt = "You are a precise data extractor. Extract only the exact target data. Output ONLY the extracted text with no explanations or thinking tags."
        user_prompt = f"Target: {req.instruction}\nInput: {req.text}"
    elif req.action == "clean":
        system_prompt = "You are a strict data cleaning utility. Clean up spaces, casing, and standard formatting. Output ONLY the cleaned result with zero conversational filler."
        user_prompt = f"Input: {req.text}"
    elif req.action == "classify":
        system_prompt = f"Classify input into one of these tags: [{req.instruction}]. Output ONLY the exact single tag name."
        user_prompt = f"Input: {req.text}"
    else:
        system_prompt = "You are SheetPulse AI. Execute the instruction directly and output only the final result."
        user_prompt = f"Instruction: {req.instruction}\nContext: {req.text}"

    client = Groq(api_key=GROQ_API_KEY)

    try:
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
                temperature=0.1,
                max_tokens=250
            )
            raw_result = completion.choices[0].message.content or ""
            clean_result = clean_ai_output(raw_result)
            return {
                "success": True,
                "result": clean_result,
                "model_used": model_name
            }
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=500, detail=f"Inference failed. Last error: {last_error}")
