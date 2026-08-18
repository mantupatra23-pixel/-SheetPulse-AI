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
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="SheetPulse AI Full-Stack", version="7.0.0")

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

# --- SQLite Database Setup ---
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

def fetch_url_text(url: str) -> str:
    try:
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            text = re.sub(r'<script.*?</script>|<style.*?</style>', '', html, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            return ' '.join(text.split())[:2500]
    except Exception as e:
        return f"Failed to load URL: {e}"

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

# --- Provider Execution Engines ---
def call_cerebras_engine(sys_prompt: str, usr_prompt: str):
    if not CEREBRAS_API_KEY:
        raise Exception("CEREBRAS_API_KEY missing")
    url = "https://api.cerebras.ai/v1/chat/completions"
    payload = {
        "model": "llama3.1-8b",
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
        "temperature": 0.1,
        "max_tokens": 250
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {CEREBRAS_API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode())
        return clean_output(res["choices"][0]["message"]["content"]), "Cerebras:Llama-3.1-8b"

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
    raise Exception("Groq busy")

def call_openrouter_engine(sys_prompt: str, usr_prompt: str):
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY missing")
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
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
        return clean_output(res["choices"][0]["message"]["content"]), "OpenRouter:Llama-3.3-70b-Free"

# --- FULL PAGE PRODUCTION FRONTEND ROUTE ---
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), SUM(total_used) FROM api_keys")
    u_count, total_exec = cur.fetchone()
    conn.close()

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>SheetPulse AI - Intelligent Spreadsheets</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <style>
        .glow-border {{
          box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.25);
        }}
      </style>
    </head>
    <body class="bg-black text-zinc-100 font-sans min-h-screen w-full flex flex-col items-center selection:bg-emerald-500 selection:text-black">
      
      <!-- Top Navbar -->
      <header class="w-full border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-zinc-900 via-zinc-950 to-black border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-black shadow-inner">
            ⚡
          </div>
          <div>
            <h1 class="text-lg font-black tracking-tight text-white flex items-center gap-1.5">
              SheetPulse <span class="bg-gradient-to-r from-emerald-400 to-green-500 bg-clip-text text-transparent">AI</span>
            </h1>
            <p class="text-[10px] text-zinc-400 tracking-wider uppercase font-mono">Enterprise Engine v7.0</p>
          </div>
        </div>
        <div class="flex items-center gap-3">
          <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/60">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Operational
          </span>
        </div>
      </header>

      <!-- Main Full-Page Content Container -->
      <main class="w-full max-w-6xl px-4 py-8 space-y-10 flex-1">
        
        <!-- Hero Section -->
        <section class="text-center space-y-4 pt-4 pb-2">
          <h2 class="text-3xl md:text-5xl font-black text-white tracking-tight max-w-3xl mx-auto leading-tight">
            Turn Any Google Sheet Into An <span class="bg-gradient-to-r from-emerald-400 via-green-400 to-emerald-500 bg-clip-text text-transparent">Autonomous AI Powerhouse</span>
          </h2>
          <p class="text-zinc-400 text-sm md:text-base max-w-2xl mx-auto">
            Extract entities, clean unstructured text, classify customer feedback, and generate complex formulas instantly with sub-second AI latency.
          </p>
          <div class="flex items-center justify-center gap-4 text-xs font-mono text-zinc-400 pt-2">
            <span class="flex items-center gap-1"><span class="text-emerald-400">●</span> {total_exec or 0} Cells Processed</span>
            <span class="flex items-center gap-1"><span class="text-emerald-400">●</span> {u_count or 0} Active Keys</span>
            <span class="flex items-center gap-1"><span class="text-emerald-400">●</span> 3 AI Clusters Connected</span>
          </div>
        </section>

        <!-- Interactive Formula Playground (Live Execution) -->
        <section class="w-full bg-zinc-950 border border-zinc-800/80 rounded-2xl p-6 md:p-8 glow-border space-y-6">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-zinc-800 pb-4">
            <div>
              <h3 class="text-lg font-bold text-white flex items-center gap-2">
                🧪 Live Formula Playground
              </h3>
              <p class="text-xs text-zinc-400">Test SheetPulse formulas directly inside the browser</p>
            </div>
            <span class="text-xs font-mono text-emerald-400 bg-zinc-900 px-3 py-1 rounded-lg border border-zinc-800">
              Active Key: sp_demo_live
            </span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="space-y-3">
              <div>
                <label class="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Select Action Formula</label>
                <select id="playAction" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-emerald-500">
                  <option value="clean">=AI_CLEAN (Standardize casing & whitespace)</option>
                  <option value="extract">=AI_EXTRACT (Regex & Entity Extractor)</option>
                  <option value="classify">=AI_CLASSIFY (Sentiment / Intent Tagging)</option>
                  <option value="fix_formula">=AI_FIX (Spreadsheet Formula Repair)</option>
                  <option value="custom">=SHEETPULSE (Custom Prompt Formula)</option>
                </select>
              </div>

              <div>
                <label class="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Input Cell Text</label>
                <textarea id="playText" rows="3" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl p-3 text-sm focus:outline-none focus:border-emerald-500 font-mono" placeholder="e.g.   mantu    patra   @   GMAIL . COM"></textarea>
              </div>

              <div id="instructionWrapper">
                <label class="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Target / Instruction</label>
                <input id="playInstruction" type="text" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-emerald-500 font-mono" placeholder="e.g. clean">
              </div>

              <!-- Black & Green Mix Button -->
              <button id="runBtn" onclick="runPlayground()" class="w-full py-3 px-6 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-zinc-950 via-emerald-950 to-black border border-emerald-500/60 hover:border-emerald-400 hover:shadow-[0_0_20px_rgba(16,185,129,0.3)] transition-all flex items-center justify-center gap-2">
                <span>⚡ Execute Formula</span>
              </button>
            </div>

            <!-- Output Display Box -->
            <div class="bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between border-b border-zinc-800/80 pb-2 mb-3">
                  <span class="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Computed Cell Output</span>
                  <span id="telemetryBadge" class="text-[11px] font-mono text-zinc-500">Ready</span>
                </div>
                <div id="outputContainer" class="text-emerald-400 font-mono text-base font-semibold break-words min-h-[100px] flex items-center">
                  Output will appear here...
                </div>
              </div>
              <div class="pt-3 border-t border-zinc-800/80 text-[11px] text-zinc-500 flex justify-between">
                <span id="providerUsed">Engine: Idle</span>
                <span id="execSpeed">Latency: 0.00s</span>
              </div>
            </div>
          </div>
        </section>

        <!-- API Key Generator & Setup Code Vault -->
        <section class="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          <!-- Key Generator Card -->
          <div class="bg-zinc-950 border border-zinc-800/80 rounded-2xl p-6 space-y-4">
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              🔑 Generate API Key
            </h3>
            <p class="text-xs text-zinc-400">Get your tenant API key to use in Google Sheets scripts.</p>
            <div class="space-y-3 pt-2">
              <input id="keyOwner" type="text" placeholder="Your Name / Business" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-emerald-500 font-mono">
              <select id="keyTier" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-emerald-500">
                <option value="free">Free Starter (100 Credits)</option>
                <option value="pro">Pro Developer (5,000 Credits)</option>
              </select>
              
              <!-- Black & Green Mix Button -->
              <button onclick="generateKey()" class="w-full py-2.5 px-4 rounded-xl font-bold text-sm text-emerald-300 bg-gradient-to-r from-zinc-900 to-black border border-emerald-600/50 hover:border-emerald-400 transition-all">
                + Create Tenant Key
              </button>
              
              <div id="keyResult" class="hidden p-3 bg-zinc-900 rounded-xl border border-emerald-800/40 text-xs font-mono text-emerald-400 break-all"></div>
            </div>
          </div>

          <!-- 1-Click Apps Script Code Vault -->
          <div class="bg-zinc-950 border border-zinc-800/80 rounded-2xl p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h3 class="text-base font-bold text-white flex items-center gap-2">
                📋 Apps Script Integration
              </h3>
              <button onclick="copyAppsScript()" class="text-xs text-emerald-400 hover:text-emerald-300 font-mono bg-zinc-900 px-3 py-1 rounded-lg border border-zinc-800">
                Copy Code
              </button>
            </div>
            <p class="text-xs text-zinc-400">Copy & paste directly into Google Sheets <code class="text-emerald-400">Extensions > Apps Script</code>.</p>
            <pre class="bg-zinc-900 p-3 rounded-xl border border-zinc-800 text-[11px] font-mono text-zinc-300 h-40 overflow-y-auto leading-relaxed">const BACKEND_URL = "https://sheetpulseai.onrender.com/api/v1/process";

function SHEETPULSE(text, instruction) {{
  return callSheetPulse(text, instruction, "custom");
}}

function AI_CLEAN(text) {{
  return callSheetPulse(text, "clean", "clean");
}}

function AI_EXTRACT(text, target) {{
  return callSheetPulse(text, target, "extract");
}}

function AI_CLASSIFY(text, categories) {{
  return callSheetPulse(text, categories, "classify");
}}

function AI_FIX(brokenFormula, goal) {{
  return callSheetPulse(brokenFormula, goal, "fix_formula");
}}

function callSheetPulse(text, inst, action) {{
  if (!text) return "";
  const payload = {{ text: String(text), instruction: String(inst || ""), action: action, api_key: "sp_demo_live" }};
  const res = UrlFetchApp.fetch(BACKEND_URL, {{
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  }});
  const json = JSON.parse(res.getContentText());
  return json.success ? json.result : "Error: " + (json.detail || "API Error");
}}</pre>
          </div>
        </section>

      </main>

      <!-- Footer -->
      <footer class="w-full border-t border-zinc-900 bg-zinc-950/60 py-6 text-center text-xs text-zinc-500 font-mono">
        &copy; 2026 SheetPulse AI Engine &bull; Built for High-Performance Spreadsheet Automations
      </footer>

      <!-- Frontend JavaScript Client -->
      <script>
        async function runPlayground() {{
          const btn = document.getElementById('runBtn');
          const output = document.getElementById('outputContainer');
          const providerBadge = document.getElementById('providerUsed');
          const speedBadge = document.getElementById('execSpeed');
          const telemetry = document.getElementById('telemetryBadge');
          
          const action = document.getElementById('playAction').value;
          const text = document.getElementById('playText').value.trim();
          const instruction = document.getElementById('playInstruction').value.trim();

          if (!text) {{
            output.innerHTML = '<span class="text-amber-400">Please provide some input text to process.</span>';
            return;
          }}

          btn.disabled = true;
          btn.innerHTML = '<span>⏳ Computing...</span>';
          output.innerHTML = '<span class="text-zinc-500 animate-pulse">Running across AI cluster...</span>';
          telemetry.innerText = 'Executing';

          const startTime = performance.now();

          try {{
            const res = await fetch('/api/v1/process', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ text: text, instruction: instruction, action: action, api_key: 'sp_demo_live' }})
            }});
            const data = await res.json();
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

            if (data.success) {{
              output.innerHTML = '<span class="text-emerald-400">' + data.result + '</span>';
              providerBadge.innerText = 'Engine: ' + (data.cached ? '⚡ In-Memory Cache' : data.provider);
              speedBadge.innerText = 'Latency: ' + elapsed + 's';
              telemetry.innerText = 'Success';
            }} else {{
              output.innerHTML = '<span class="text-red-400">Error: ' + (data.detail || 'Failed') + '</span>';
              telemetry.innerText = 'Failed';
            }}
          }} catch (err) {{
            output.innerHTML = '<span class="text-red-400">Network Error: ' + err.message + '</span>';
          }} finally {{
            btn.disabled = false;
            btn.innerHTML = '<span>⚡ Execute Formula</span>';
          }}
        }}

        async function generateKey() {{
          const owner = document.getElementById('keyOwner').value.trim() || 'User';
          const tier = document.getElementById('keyTier').value;
          const box = document.getElementById('keyResult');
          
          box.classList.remove('hidden');
          box.innerText = 'Generating key...';

          try {{
            const res = await fetch('/api/v1/keys/new', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ owner_name: owner, tier: tier }})
            }});
            const data = await res.json();
            box.innerHTML = '✅ <strong>Your API Key:</strong><br><code class="text-white select-all">' + data.api_key + '</code><br><span class="text-zinc-400">Credits: ' + data.credits + ' | Tier: ' + data.tier + '</span>';
          }} catch (e) {{
            box.innerText = 'Error generating key.';
          }}
        }}

        function copyAppsScript() {{
          const code = `const BACKEND_URL = "https://sheetpulseai.onrender.com/api/v1/process";

function SHEETPULSE(text, instruction) { return callSheetPulse(text, instruction, "custom"); }
function AI_CLEAN(text) { return callSheetPulse(text, "clean", "clean"); }
function AI_EXTRACT(text, target) { return callSheetPulse(text, target, "extract"); }
function AI_CLASSIFY(text, categories) { return callSheetPulse(text, categories, "classify"); }
function AI_FIX(brokenFormula, goal) { return callSheetPulse(brokenFormula, goal, "fix_formula"); }

function callSheetPulse(text, inst, action) {
  if (!text) return "";
  const payload = { text: String(text), instruction: String(inst || ""), action: action, api_key: "sp_demo_live" };
  const res = UrlFetchApp.fetch(BACKEND_URL, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });
  const json = JSON.parse(res.getContentText());
  return json.success ? json.result : "Error: " + (json.detail || "API Error");
}`;
          navigator.clipboard.writeText(code);
          alert("Apps Script code copied to clipboard!");
        }}
      </script>
    </body>
    </html>
    """

# --- API KEY ENDPOINTS ---
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

# --- PROCESS CELL ENDPOINT ---
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
    text_content = req.text

    if act == "scrape":
        scraped_data = fetch_url_text(req.text.strip())
        sys = "Analyze the web page and extract the requested answer directly. Output ONLY the answer."
        usr = f"Question: {req.instruction}\nContent:\n{scraped_data}"
    elif act == "clean":
        sys = "Standardize formatting, fix broken spaces/casing. Output ONLY cleaned result."
        usr = f"Input: {text_content}"
    elif act == "extract":
        sys = "Extract the requested entity. Output ONLY the extracted text."
        usr = f"Target: {req.instruction}\nText: {text_content}"
    elif act == "classify":
        sys = f"Classify input into ONE tag from: [{req.instruction}]. Output ONLY the exact tag name."
        usr = f"Input: {text_content}"
    elif act == "fix_formula":
        sys = "Analyze broken spreadsheet formula and fix it. Output ONLY valid formula starting with '='."
        usr = f"Broken Formula: {text_content}\nContext: {req.instruction}"
    elif act == "formula":
        sys = "Generate a Google Sheets formula starting with '='. Output ONLY the formula."
        usr = f"Requirement: {req.instruction}\nContext: {text_content}"
    else:
        sys = "Execute the instruction directly. Output ONLY the direct final answer."
        usr = f"Instruction: {req.instruction}\nContext: {text_content}"

    async with CONCURRENCY_LIMIT:
        result, provider = None, None
        try:
            result, provider = call_cerebras_engine(sys, usr)
        except Exception:
            try:
                result, provider = call_groq_engine(sys, usr)
            except Exception:
                try:
                    result, provider = call_openrouter_engine(sys, usr)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"All clusters failed: {e}")

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
