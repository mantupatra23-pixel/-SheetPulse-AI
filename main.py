from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SheetPulse AI API")

# Google Sheets / Apps Script aur Web Dashboard se request allow karne ke liye CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    text: str
    instruction: str = "clean"

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "SheetPulse AI Backend",
        "version": "1.0.0"
    }

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Empty input cell")
    
    # Test response taaki deployment verify ho sake
    return {
        "success": True,
        "input": req.text,
        "result": f"[SheetPulse Processed] {req.text.strip()}"
    }
