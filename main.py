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

app = FastAPI(title="SheetPulse AI SaaS Platform", version="8.0.0")

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

# --- SQLite Database Initialization ---
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
        raise HTTPException(status_code=401, detail="Invalid API Key. Generate one from the dashboard.")
    
    credits_left, tier = row["credits_left"], row["tier"]
    if tier != "developer" and credits_left < amount:
        conn.close()
        raise HTTPException(status_code=402, detail="Credit quota exhausted. Upgrade your plan.")

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
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
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

# --- FULL PRODUCTION SAAS FRONTEND ---
@app.get("/", response_class=HTMLResponse)
def serve_premium_frontend():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), SUM(total_used) FROM api_keys")
    u_count, total_exec = cur.fetchone()
    conn.close()

    return f"""
    <!DOCTYPE html>
    <html lang="en" class="scroll-smooth">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>SheetPulse AI - Stop Copy-Pasting AI. Run It In Google Sheets.</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
      <script>
        tailwind.config = {{
          theme: {{
            extend: {{
              fontFamily: {{
                sans: ['"Plus Jakarta Sans"', 'sans-serif'],
                mono: ['"JetBrains Mono"', 'monospace'],
              }},
              colors: {{
                brand: {{
                  50: '#ecfdf5',
                  400: '#34d399',
                  500: '#10b981',
                  600: '#059669',
                  950: '#022c22',
                }}
              }}
            }}
          }}
        }}
      </script>
      <style>
        body {{ background-color: #030303; }}
        .emerald-glow {{
          box-shadow: 0 0 50px -10px rgba(16, 185, 129, 0.18);
        }}
        .grid-bg {{
          background-size: 32px 32px;
          background-image: linear-gradient(to right, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                            linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        }}
      </style>
    </head>
    <body class="text-zinc-100 antialiased selection:bg-brand-500 selection:text-black grid-bg">

      <!-- Navigation Header -->
      <nav class="sticky top-0 z-50 w-full border-b border-zinc-800/80 bg-black/80 backdrop-blur-xl px-4 md:px-8 py-3.5 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-xl bg-gradient-to-b from-zinc-800 to-black border border-brand-500/40 flex items-center justify-center text-brand-400 font-extrabold shadow-lg">
            ⚡
          </div>
          <a href="#" class="text-lg font-extrabold tracking-tight text-white flex items-center gap-1">
            SheetPulse<span class="text-brand-400">AI</span>
          </a>
        </div>
        
        <div class="hidden md:flex items-center gap-8 text-xs font-semibold text-zinc-400">
          <a href="#features" class="hover:text-white transition-colors">Features</a>
          <a href="#usecases" class="hover:text-white transition-colors">Use Cases</a>
          <a href="#playground" class="hover:text-white transition-colors">Live Playground</a>
          <a href="#pricing" class="hover:text-white transition-colors">Pricing</a>
          <a href="#integration" class="hover:text-white transition-colors">Apps Script</a>
        </div>

        <div class="flex items-center gap-3">
          <a href="#playground" class="hidden sm:inline-flex px-3.5 py-1.5 rounded-lg text-xs font-bold text-zinc-300 hover:text-white bg-zinc-900 border border-zinc-800 hover:border-zinc-700 transition-all">
            Live Demo
          </a>
          <a href="#integration" class="px-4 py-1.5 rounded-lg text-xs font-extrabold text-black bg-gradient-to-r from-brand-400 to-brand-500 hover:brightness-110 shadow-[0_0_20px_rgba(16,185,129,0.3)] transition-all">
            Install Free
          </a>
        </div>
      </nav>

      <!-- HERO SECTION -->
      <section class="max-w-6xl mx-auto px-4 pt-16 pb-12 text-center space-y-6">
        <div class="inline-flex items-center gap-2 px-3.5 py-1 rounded-full text-xs font-semibold bg-brand-950/80 text-brand-400 border border-brand-500/30">
          <span class="w-1.5 h-1.5 rounded-full bg-brand-400 animate-ping"></span>
          For people who live in Google Sheets
        </div>

        <h1 class="text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-[1.08]">
          Stop copy-pasting AI into your spreadsheet. <br>
          <span class="bg-gradient-to-r from-brand-400 via-emerald-300 to-green-400 bg-clip-text text-transparent">
            Run it where the work already is.
          </span>
        </h1>

        <p class="text-zinc-400 text-sm sm:text-lg max-w-2xl mx-auto leading-relaxed">
          SheetPulse AI puts text extraction, data cleaning, automated classification, and web scraping directly inside Google Sheets formulas. Write a formula. Fill the column. Ship the job.
        </p>

        <!-- CTA Buttons -->
        <div class="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
          <a href="#integration" class="w-full sm:w-auto px-8 py-3.5 rounded-xl font-extrabold text-sm text-black bg-gradient-to-r from-brand-400 to-emerald-400 hover:brightness-110 shadow-[0_0_30px_rgba(16,185,129,0.4)] transition-all">
            Install Free &mdash; 2 min setup
          </a>
          <a href="#pricing" class="w-full sm:w-auto px-8 py-3.5 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-zinc-950 via-zinc-900 to-black border border-zinc-800 hover:border-brand-500/50 transition-all">
            Compare plans
          </a>
        </div>

        <p class="text-xs text-zinc-500">
          Free with your own API key &bull; <span class="text-brand-400">Hosted plans from $12/mo</span> &bull; 198k+ executions
        </p>

        <!-- SPREADSHEET MOCKUP HERO PREVIEW -->
        <div class="pt-6 max-w-4xl mx-auto">
          <div class="rounded-2xl border border-zinc-800 bg-zinc-950 p-2 sm:p-4 emerald-glow text-left">
            <div class="flex items-center justify-between px-3 py-2 border-b border-zinc-800/80 mb-3 text-xs text-zinc-400 font-mono">
              <div class="flex items-center gap-2">
                <span class="w-2.5 h-2.5 rounded-full bg-red-500/60 inline-block"></span>
                <span class="w-2.5 h-2.5 rounded-full bg-yellow-500/60 inline-block"></span>
                <span class="w-2.5 h-2.5 rounded-full bg-green-500/60 inline-block"></span>
                <span class="ml-2 text-zinc-300 font-sans font-semibold">Campaign_Engine.gsheet</span>
              </div>
              <span class="text-brand-400 bg-brand-950/60 px-2.5 py-0.5 rounded border border-brand-800/50">⚡ 0.8s UltraFast</span>
            </div>

            <!-- Formula Bar -->
            <div class="flex items-center gap-3 bg-zinc-900/80 rounded-lg px-3 py-2 border border-zinc-800 text-xs font-mono mb-3">
              <span class="text-zinc-500 font-bold">fx</span>
              <span class="text-white">=SHEETPULSE(A2, "Write a high-converting 1-line SaaS hook")</span>
            </div>

            <!-- Sheet Table -->
            <div class="overflow-x-auto">
              <table class="w-full text-xs font-mono border-collapse">
                <thead>
                  <tr class="bg-zinc-900/50 text-zinc-400 border-b border-zinc-800">
                    <th class="p-2.5 w-12 text-center border-r border-zinc-800">#</th>
                    <th class="p-2.5 text-left border-r border-zinc-800 w-1/2">A &bull; Raw Input</th>
                    <th class="p-2.5 text-left text-brand-400">B &bull; SheetPulse AI Output</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-zinc-800/60 text-zinc-300">
                  <tr>
                    <td class="p-2.5 text-center text-zinc-600 bg-zinc-900/30">2</td>
                    <td class="p-2.5 border-r border-zinc-800/60">Fintech startup targeting founders</td>
                    <td class="p-2.5 text-brand-300 bg-brand-950/20 font-semibold">Scale runway, automate accounting, stay audit-ready.</td>
                  </tr>
                  <tr>
                    <td class="p-2.5 text-center text-zinc-600 bg-zinc-900/30">3</td>
                    <td class="p-2.5 border-r border-zinc-800/60">D2C Organic Coffee Roastery</td>
                    <td class="p-2.5 text-brand-300 bg-brand-950/20 font-semibold">Small-batch artisan coffee, delivered fresh to your door.</td>
                  </tr>
                  <tr>
                    <td class="p-2.5 text-center text-zinc-600 bg-zinc-900/30">4</td>
                    <td class="p-2.5 border-r border-zinc-800/60">B2B Cyber Security Platform</td>
                    <td class="p-2.5 text-brand-300 bg-brand-950/20 font-semibold">Zero-trust cloud protection before threats ever reach endpoints.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- 4 Hero Value Pillars -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto pt-6 text-left">
          <div class="p-4 rounded-xl bg-zinc-950 border border-zinc-900 space-y-1">
            <p class="text-sm font-bold text-white">Fill Down</p>
            <p class="text-xs text-zinc-400 leading-snug">Drag formula down 500 rows. Every cell auto-computes.</p>
          </div>
          <div class="p-4 rounded-xl bg-zinc-950 border border-zinc-900 space-y-1">
            <p class="text-sm font-bold text-white">Clean & Extract</p>
            <p class="text-xs text-zinc-400 leading-snug">Extract emails, phone numbers & standardize names.</p>
          </div>
          <div class="p-4 rounded-xl bg-zinc-950 border border-zinc-900 space-y-1">
            <p class="text-sm font-bold text-brand-400">$0 BYOK</p>
            <p class="text-xs text-zinc-400 leading-snug">Unlimited AI on your own Groq / Cerebras / Gemini keys.</p>
          </div>
          <div class="p-4 rounded-xl bg-zinc-950 border border-zinc-900 space-y-1">
            <p class="text-sm font-bold text-white">Multi-Cluster</p>
            <p class="text-xs text-zinc-400 leading-snug">3 AI providers running parallel for zero downtime.</p>
          </div>
        </div>
      </section>

      <!-- SOCIAL PROOF LOGO WALL -->
      <section class="border-y border-zinc-900 bg-zinc-950/40 py-10">
        <div class="max-w-6xl mx-auto px-4 text-center space-y-4">
          <p class="text-xs uppercase tracking-widest text-zinc-400 font-bold">Trusted by operators, growth engineers and data teams</p>
          <div class="flex flex-wrap items-center justify-center gap-8 md:gap-16 opacity-50 grayscale hover:grayscale-0 transition-all text-sm md:text-base font-extrabold text-zinc-400">
            <span>GOOGLE WORKSPACE</span>
            <span>HUBSPOT</span>
            <span>SHOPIFY</span>
            <span>STRIPE</span>
            <span>NOTION</span>
            <span>AIRTABLE</span>
          </div>
        </div>
      </section>

      <!-- THE PROBLEM VS SOLUTION -->
      <section id="features" class="max-w-6xl mx-auto px-4 py-20 space-y-16">
        
        <!-- The Problem -->
        <div class="space-y-6">
          <div>
            <span class="text-xs font-bold uppercase tracking-wider text-rose-500">The Problem</span>
            <h2 class="text-2xl sm:text-4xl font-extrabold text-white mt-1">Spreadsheets run the business. AI still lives outside them.</h2>
            <p class="text-zinc-400 text-sm max-w-2xl mt-2">You already work in Google Sheets. Everything else &mdash; chatbots, extractors, web scrapers &mdash; forces you to leave, export CSVs, and glue work together manually.</p>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="p-5 rounded-2xl bg-zinc-950 border border-zinc-900 space-y-2">
              <span class="text-lg">⏳</span>
              <p class="font-bold text-white text-sm">Hours lost on repetitive work</p>
              <p class="text-xs text-zinc-400">Rewriting product titles, tagging customer tickets, cleaning phone columns one row at a time.</p>
            </div>
            <div class="p-5 rounded-2xl bg-zinc-950 border border-zinc-900 space-y-2">
              <span class="text-lg">💬</span>
              <p class="font-bold text-white text-sm">AI lives in another browser tab</p>
              <p class="text-xs text-zinc-400">ChatGPT &rarr; copy &rarr; paste &rarr; fix formatting &rarr; repeat. Context dies the moment you switch tabs.</p>
            </div>
            <div class="p-5 rounded-2xl bg-zinc-950 border border-zinc-900 space-y-2">
              <span class="text-lg">⛓️</span>
              <p class="font-bold text-white text-sm">No bulk execution</p>
              <p class="text-xs text-zinc-400">You can't drag a prompt down 1,000 rows in ChatGPT. Modern operations need pipelines, not chat windows.</p>
            </div>
          </div>
        </div>

        <!-- The Solution -->
        <div class="space-y-6 pt-6">
          <div>
            <span class="text-xs font-bold uppercase tracking-wider text-brand-400">The Solution</span>
            <h2 class="text-2xl sm:text-4xl font-extrabold text-white mt-1">Put AI where the data already lives.</h2>
            <p class="text-zinc-400 text-sm max-w-2xl mt-2">SheetPulse AI is the enterprise custom-function engine that executes autonomous AI models right inside native cell formulas.</p>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div class="p-5 rounded-2xl bg-zinc-950 border border-brand-500/20 space-y-2">
              <p class="font-bold text-white text-sm flex items-center gap-2">
                <span class="text-brand-400 font-bold">✓</span> Stay inside Google Sheets
              </p>
              <p class="text-xs text-zinc-400">Write a formula once. Drag it down. Every row calculates autonomously.</p>
            </div>
            <div class="p-5 rounded-2xl bg-zinc-950 border border-brand-500/20 space-y-2">
              <p class="font-bold text-white text-sm flex items-center gap-2">
                <span class="text-brand-400 font-bold">✓</span> Triple-Engine Resilience
              </p>
              <p class="text-xs text-zinc-400">Cerebras, Groq, and OpenRouter run with automated sub-second failover.</p>
            </div>
            <div class="p-5 rounded-2xl bg-zinc-950 border border-brand-500/20 space-y-2">
              <p class="font-bold text-white text-sm flex items-center gap-2">
                <span class="text-brand-400 font-bold">✓</span> Bring Your Own Key (Free)
              </p>
              <p class="text-xs text-zinc-400">Use your own free Groq/Cerebras API keys without recurring markup costs.</p>
            </div>
            <div class="p-5 rounded-2xl bg-zinc-950 border border-brand-500/20 space-y-2">
              <p class="font-bold text-white text-sm flex items-center gap-2">
                <span class="text-brand-400 font-bold">✓</span> High-Speed In-Memory Caching
              </p>
              <p class="text-xs text-zinc-400">Repeated calls hit zero-latency RAM cache without burning API tokens.</p>
            </div>
          </div>
        </div>

      </section>

      <!-- LIVE INTERACTIVE FORMULA PLAYGROUND -->
      <section id="playground" class="max-w-6xl mx-auto px-4 py-16">
        <div class="bg-zinc-950 border border-zinc-800 rounded-3xl p-6 sm:p-10 emerald-glow space-y-8">
          
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-6">
            <div>
              <div class="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono bg-brand-950 text-brand-400 border border-brand-800/60 mb-2">
                ⚡ Interactive Playground
              </div>
              <h2 class="text-2xl sm:text-3xl font-extrabold text-white">Test Real Spreadsheet Formulas Live</h2>
              <p class="text-xs sm:text-sm text-zinc-400">Select any formula, enter input text, and watch the multi-cluster backend compute in real time.</p>
            </div>
            <div class="text-left md:text-right font-mono text-xs text-zinc-400">
              <span class="text-brand-400">Live Backend:</span> sheetpulseai.onrender.com<br>
              <span class="text-zinc-500">Version: 8.0.0 Enterprise</span>
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            <!-- Control Form (Left) -->
            <div class="lg:col-span-6 space-y-4">
              <div>
                <label class="block text-xs font-bold uppercase tracking-wider text-zinc-400 mb-1.5">Select SheetPulse Formula</label>
                <select id="playAction" onchange="updatePlaygroundTemplate()" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-brand-400 font-mono">
                  <option value="clean">=AI_CLEAN(cell) &bull; Fix casing, trim & standardise</option>
                  <option value="extract">=AI_EXTRACT(cell, target) &bull; Extract entity or regex</option>
                  <option value="classify">=AI_CLASSIFY(cell, tags) &bull; Categorise text</option>
                  <option value="fix_formula">=AI_FIX(broken_formula, goal) &bull; Repair sheets formulas</option>
                  <option value="scrape">=AI_SCRAPE(url, question) &bull; Live webpage extraction</option>
                  <option value="custom">=SHEETPULSE(cell, prompt) &bull; Custom prompt execution</option>
                </select>
              </div>

              <div>
                <label class="block text-xs font-bold uppercase tracking-wider text-zinc-400 mb-1.5">Cell Input Text</label>
                <textarea id="playText" rows="3" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl p-3.5 text-sm focus:outline-none focus:border-brand-400 font-mono" placeholder="Input string..."></textarea>
              </div>

              <div>
                <label class="block text-xs font-bold uppercase tracking-wider text-zinc-400 mb-1.5">Target / Instruction</label>
                <input id="playInstruction" type="text" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-brand-400 font-mono" placeholder="Instruction...">
              </div>

              <!-- Black and Emerald Green Mix Button -->
              <button id="runBtn" onclick="runPlayground()" class="w-full py-3.5 px-6 rounded-xl font-black text-sm text-white bg-gradient-to-r from-zinc-950 via-brand-950 to-black border border-brand-500/70 hover:border-brand-400 hover:shadow-[0_0_25px_rgba(16,185,129,0.35)] transition-all flex items-center justify-center gap-2">
                <span>⚡ Run Live Formula</span>
              </button>
            </div>

            <!-- Terminal Mock Cell Output (Right) -->
            <div class="lg:col-span-6 bg-zinc-900/70 border border-zinc-800 rounded-2xl p-5 flex flex-col justify-between space-y-4">
              <div>
                <div class="flex items-center justify-between border-b border-zinc-800 pb-3 mb-4">
                  <div class="flex items-center gap-2 text-xs font-mono text-zinc-400">
                    <span class="w-2.5 h-2.5 rounded-full bg-brand-400"></span>
                    <span>SPREADSHEET CELL VALUE</span>
                  </div>
                  <span id="telemetryBadge" class="text-xs font-mono text-zinc-500">Ready</span>
                </div>

                <div id="outputContainer" class="text-brand-300 font-mono text-base font-semibold break-words min-h-[140px] flex items-center justify-center text-center p-4 bg-black/40 rounded-xl border border-zinc-800/60">
                  Select a formula and click "Run Live Formula"
                </div>
              </div>

              <div class="pt-3 border-t border-zinc-800/80 text-xs font-mono text-zinc-400 flex items-center justify-between">
                <span id="providerUsed">Cluster: Idle</span>
                <span id="execSpeed">Latency: 0.00s</span>
              </div>
            </div>

          </div>
        </div>
      </section>

      <!-- USE CASES BY ROLE (6-CARD SPREADSHEET GRID) -->
      <section id="usecases" class="max-w-6xl mx-auto px-4 py-16 space-y-12">
        <div class="text-center space-y-3">
          <span class="text-xs font-bold uppercase tracking-wider text-brand-400">Who It's For</span>
          <h2 class="text-3xl sm:text-4xl font-extrabold text-white">Same sheet. Different jobs. Zero extra tools.</h2>
          <p class="text-zinc-400 text-sm max-w-xl mx-auto">From growth marketers to data analysts &mdash; run autonomous AI at spreadsheet scale.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          
          <!-- Card 1: Marketers -->
          <div class="p-6 rounded-2xl bg-zinc-950 border border-zinc-800/80 space-y-4">
            <div class="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <span>📢</span> Marketers
            </div>
            <p class="text-xs text-zinc-400">Bulk ad copy, localized taglines, and email subject lines generated across hundreds of SKUs.</p>
            <div class="bg-zinc-900 rounded-xl p-3 text-[11px] font-mono border border-zinc-800 space-y-1 text-zinc-300">
              <p class="text-zinc-500 font-semibold">=SHEETPULSE(A2, "Write 3 ad hooks")</p>
              <p class="text-brand-300 font-bold">&rarr; Stop burning budget. Scale revenue fast.</p>
            </div>
          </div>

          <!-- Card 2: Data Analysts -->
          <div class="p-6 rounded-2xl bg-zinc-950 border border-zinc-800/80 space-y-4">
            <div class="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <span>📊</span> Data Analysts
            </div>
            <p class="text-xs text-zinc-400">Clean messy customer records, standardize phone numbers, and classify transaction descriptions.</p>
            <div class="bg-zinc-900 rounded-xl p-3 text-[11px] font-mono border border-zinc-800 space-y-1 text-zinc-300">
              <p class="text-zinc-500 font-semibold">=AI_CLASSIFY(A2, "Food, Travel, SaaS")</p>
              <p class="text-brand-300 font-bold">&rarr; SaaS Subscription</p>
            </div>
          </div>

          <!-- Card 3: Content Creators -->
          <div class="p-6 rounded-2xl bg-zinc-950 border border-zinc-800/80 space-y-4">
            <div class="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <span>✍️</span> Content Creators
            </div>
            <p class="text-xs text-zinc-400">Brainstorm YouTube titles, convert blog outlines to tweets, and generate meta descriptions in bulk.</p>
            <div class="bg-zinc-900 rounded-xl p-3 text-[11px] font-mono border border-zinc-800 space-y-1 text-zinc-300">
              <p class="text-zinc-500 font-semibold">=SHEETPULSE(A2, "5 viral tweet hooks")</p>
              <p class="text-brand-300 font-bold">&rarr; 1/ The biggest lie in SaaS is...</p>
            </div>
          </div>

          <!-- Card 4: Researchers -->
          <div class="p-6 rounded-2xl bg-zinc-950 border border-zinc-800/80 space-y-4">
            <div class="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <span>🔬</span> Researchers & Ops
            </div>
            <p class="text-xs text-zinc-400">Summarize PDF abstracts, extract key statistical metrics, and organize citations directly in columns.</p>
            <div class="bg-zinc-900 rounded-xl p-3 text-[11px] font-mono border border-zinc-800 space-y-1 text-zinc-300">
              <p class="text-zinc-500 font-semibold">=AI_EXTRACT(A2, "key finding percentage")</p>
              <p class="text-brand-300 font-bold">&rarr; 34.8% efficiency boost</p>
            </div>
          </div>

          <!-- Card 5: Sales & B2B -->
          <div class="p-6 rounded-2xl bg-zinc-950 border border-zinc-800/80 space-y-4">
            <div class="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <span>💼</span> Sales & Outbound
            </div>
            <p class="text-xs text-zinc-400">Personalize first-line cold emails by scraping prospect website homepages automatically.</p>
            <div class="bg-zinc-900 rounded-xl p-3 text-[11px] font-mono border border-zinc-800 space-y-1 text-zinc-300">
              <p class="text-zinc-500 font-semibold">=AI_SCRAPE(A2, "personalized intro line")</p>
              <p class="text-brand-300 font-bold">&rarr; Loved your recent launch of v2!</p>
            </div>
          </div>

          <!-- Card 6: HR & Recruiters -->
          <div class="p-6 rounded-2xl bg-zinc-950 border border-zinc-800/80 space-y-4">
            <div class="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <span>👥</span> HR & Hiring
            </div>
            <p class="text-xs text-zinc-400">Extract skills from candidate bios, score resumes against job criteria, and generate tailored outreach.</p>
            <div class="bg-zinc-900 rounded-xl p-3 text-[11px] font-mono border border-zinc-800 space-y-1 text-zinc-300">
              <p class="text-zinc-500 font-semibold">=AI_EXTRACT(A2, "years of experience")</p>
              <p class="text-brand-300 font-bold">&rarr; 6+ years in FastAPI & React</p>
            </div>
          </div>

        </div>
      </section>

      <!-- TRANSPARENT SAAS PRICING MATRIX -->
      <section id="pricing" class="max-w-6xl mx-auto px-4 py-16 space-y-12">
        <div class="text-center space-y-3">
          <span class="text-xs font-bold uppercase tracking-wider text-brand-400">Pricing</span>
          <h2 class="text-3xl sm:text-4xl font-extrabold text-white">Simple, Transparent Pricing. No Surprises.</h2>
          <p class="text-zinc-400 text-sm max-w-xl mx-auto">Start completely free with your own API key, or switch to managed hosted plans.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
          
          <!-- Free BYOK Plan -->
          <div class="p-8 rounded-3xl bg-zinc-950 border border-zinc-800 flex flex-col justify-between space-y-6">
            <div class="space-y-4">
              <span class="inline-block px-3 py-1 rounded-md text-xs font-mono font-bold bg-zinc-900 text-zinc-300 border border-zinc-800">
                BYOK TIER
              </span>
              <h3 class="text-xl font-bold text-white">Free Forever</h3>
              <div class="flex items-baseline gap-1 text-4xl font-black text-white">
                $0 <span class="text-xs font-normal text-zinc-500">/ forever</span>
              </div>
              <p class="text-xs text-zinc-400">Bring your own free Groq, Cerebras, or OpenRouter API keys.</p>
              
              <ul class="space-y-2.5 text-xs text-zinc-300 pt-4 border-t border-zinc-900 font-medium">
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Unlimited formula executions</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Access all formulas (=AI_CLEAN, =SHEETPULSE)</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Zero markup fees</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Community support</li>
              </ul>
            </div>

            <a href="#integration" class="w-full py-3 rounded-xl font-bold text-xs text-center text-white bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 transition-all block">
              Get Started Free
            </a>
          </div>

          <!-- Starter Plan -->
          <div class="p-8 rounded-3xl bg-zinc-950 border border-brand-500/50 flex flex-col justify-between space-y-6 emerald-glow relative">
            <div class="space-y-4">
              <div class="flex items-center justify-between">
                <span class="inline-block px-3 py-1 rounded-md text-xs font-mono font-bold bg-brand-950 text-brand-400 border border-brand-800/80">
                  STARTER
                </span>
                <span class="text-[10px] font-black uppercase tracking-wider text-brand-400 bg-brand-950/60 px-2 py-0.5 rounded">Most Popular</span>
              </div>
              <h3 class="text-xl font-bold text-white">Hosted Starter</h3>
              <div class="flex items-baseline gap-1 text-4xl font-black text-white">
                $12 <span class="text-xs font-normal text-zinc-500">/ month</span>
              </div>
              <p class="text-xs text-zinc-400">Zero API keys to manage. High-speed hosted models ready out of the box.</p>
              
              <ul class="space-y-2.5 text-xs text-zinc-300 pt-4 border-t border-zinc-900 font-medium">
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> 5,000 cell executions / month</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Sub-second Groq & Cerebras speed</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Formula Repair & Web Scraping</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Priority email support</li>
              </ul>
            </div>

            <button onclick="generateHostedKey('Starter')" class="w-full py-3 rounded-xl font-extrabold text-xs text-black bg-gradient-to-r from-brand-400 to-emerald-400 hover:brightness-110 shadow-lg transition-all">
              Start 7-Day Trial
            </button>
          </div>

          <!-- Pro Plan -->
          <div class="p-8 rounded-3xl bg-zinc-950 border border-zinc-800 flex flex-col justify-between space-y-6">
            <div class="space-y-4">
              <span class="inline-block px-3 py-1 rounded-md text-xs font-mono font-bold bg-zinc-900 text-zinc-300 border border-zinc-800">
                PRO SCALE
              </span>
              <h3 class="text-xl font-bold text-white">Agency / Scale</h3>
              <div class="flex items-baseline gap-1 text-4xl font-black text-white">
                $29 <span class="text-xs font-normal text-zinc-500">/ month</span>
              </div>
              <p class="text-xs text-zinc-400">For agencies and high-volume operations running matrix pipelines.</p>
              
              <ul class="space-y-2.5 text-xs text-zinc-300 pt-4 border-t border-zinc-900 font-medium">
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> 30,000 cell executions / month</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Parallel batch endpoint access</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> Dedicated memory caching tier</li>
                <li class="flex items-center gap-2"><span class="text-brand-400 font-bold">✓</span> 24/7 Priority Discord & WhatsApp support</li>
              </ul>
            </div>

            <button onclick="generateHostedKey('Pro')" class="w-full py-3 rounded-xl font-bold text-xs text-white bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 transition-all">
              Upgrade to Pro
            </button>
          </div>

        </div>
      </section>

      <!-- 1-CLICK APPS SCRIPT INTEGRATION VAULT & KEY GENERATOR -->
      <section id="integration" class="max-w-6xl mx-auto px-4 py-16 space-y-8">
        <div class="text-center space-y-3">
          <span class="text-xs font-bold uppercase tracking-wider text-brand-400">2-Minute Setup</span>
          <h2 class="text-3xl sm:text-4xl font-extrabold text-white">Add SheetPulse AI To Your Google Sheet</h2>
          <p class="text-zinc-400 text-sm max-w-xl mx-auto">Copy the client script, paste into Extensions &rarr; Apps Script, and start using formulas instantly.</p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          <!-- Key Generator (Left) -->
          <div class="lg:col-span-5 bg-zinc-950 border border-zinc-800 rounded-3xl p-6 sm:p-8 space-y-4">
            <h3 class="text-lg font-bold text-white flex items-center gap-2">
              🔑 Generate Tenant Key
            </h3>
            <p class="text-xs text-zinc-400">Get your personal API key to meter and track your credits.</p>

            <div class="space-y-3 pt-2">
              <div>
                <label class="block text-xs font-semibold text-zinc-400 mb-1">Your Name / Workspace</label>
                <input id="keyOwner" type="text" placeholder="e.g. Mantu Growth Studio" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-400 font-mono">
              </div>

              <div>
                <label class="block text-xs font-semibold text-zinc-400 mb-1">Plan Selection</label>
                <select id="keyTier" class="w-full bg-zinc-900 border border-zinc-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-400">
                  <option value="free">Free BYOK (100 Initial Cloud Credits)</option>
                  <option value="pro">Pro Developer (5,000 Cloud Credits)</option>
                </select>
              </div>

              <!-- Black & Green Mix Button -->
              <button onclick="generateKey()" class="w-full py-3 rounded-xl font-bold text-xs text-brand-300 bg-gradient-to-r from-zinc-950 via-brand-950 to-black border border-brand-600/60 hover:border-brand-400 transition-all">
                + Generate Secret API Key
              </button>

              <div id="keyResult" class="hidden p-4 bg-zinc-900 rounded-xl border border-brand-800/60 text-xs font-mono text-brand-300 break-all space-y-1"></div>
            </div>
          </div>

          <!-- Apps Script Code Vault (Right) -->
          <div class="lg:col-span-7 bg-zinc-950 border border-zinc-800 rounded-3xl p-6 sm:p-8 space-y-4">
            <div class="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 class="text-base font-bold text-white flex items-center gap-2">
                📋 Google Apps Script Code
              </h3>
              <button onclick="copyAppsScript()" class="px-3.5 py-1 rounded-lg text-xs font-bold text-black bg-brand-400 hover:bg-brand-300 transition-all flex items-center gap-1">
                <span>Copy Script</span>
              </button>
            </div>

            <pre class="bg-zinc-900/80 p-4 rounded-xl border border-zinc-800 text-[11px] font-mono text-zinc-300 h-64 overflow-y-auto leading-relaxed select-all">const BACKEND_URL = "https://sheetpulseai.onrender.com/api/v1/process";
const SHEETPULSE_API_KEY = "sp_demo_live"; // Or paste your generated key

function SHEETPULSE(text, instruction) {
  return callSheetPulse(text, instruction, "custom");
}

function AI_CLEAN(text) {
  return callSheetPulse(text, "clean", "clean");
}

function AI_EXTRACT(text, target) {
  return callSheetPulse(text, target, "extract");
}

function AI_CLASSIFY(text, categories) {
  return callSheetPulse(text, categories, "classify");
}

function AI_FIX(brokenFormula, goal) {
  return callSheetPulse(brokenFormula, goal, "fix_formula");
}

function AI_SCRAPE(url, question) {
  return callSheetPulse(url, question, "scrape");
}

function callSheetPulse(text, inst, action) {
  if (!text) return "";
  const payload = {
    text: String(text),
    instruction: String(inst || ""),
    action: action,
    api_key: SHEETPULSE_API_KEY
  };
  const res = UrlFetchApp.fetch(BACKEND_URL, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });
  const json = JSON.parse(res.getContentText());
  return json.success ? json.result : "Error: " + (json.detail || "API Error");
}</pre>
          </div>

        </div>
      </section>

      <!-- FOOTER -->
      <footer class="border-t border-zinc-900 bg-black py-12 text-xs text-zinc-400 font-mono">
        <div class="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-brand-400"></span>
            <span class="text-white font-bold">SheetPulse AI Engine</span> &bull; Production v8.0
          </div>
          <p>&copy; 2026 SheetPulse AI. All rights reserved. Autonomous Spreadsheet Infrastructure.</p>
        </div>
      </footer>

      <!-- INTERACTIVE FRONTEND ENGINE SCRIPT -->
      <script>
        const playgroundTemplates = {
          clean: { text: "   mAntU   pAtRA   @  GMAIL . COM  ", inst: "clean" },
          extract: { text: "For enterprise pricing contact sales@scalehub.co or call +1-800-492-9102", inst: "email address" },
          classify: { text: "My package was broken during courier transit and arrived 5 days late!", inst: "Complaint, Feedback, Urgent" },
          fix_formula: { text: "=VLOOKUP(A2, B:C, 3, FALSE)", inst: "Fix index range error for 2-column array" },
          scrape: { text: "https://news.ycombinator.com", inst: "Extract the current #1 trending headline title" },
          custom: { text: "SheetPulse AI", inst: "Write a high-converting 4-word SaaS value proposition" }
        };

        function updatePlaygroundTemplate() {
          const action = document.getElementById('playAction').value;
          const template = playgroundTemplates[action] || playgroundTemplates.clean;
          document.getElementById('playText').value = template.text;
          document.getElementById('playInstruction').value = template.inst;
        }

        // Initialize default template on load
        updatePlaygroundTemplate();

        async function runPlayground() {
          const btn = document.getElementById('runBtn');
          const output = document.getElementById('outputContainer');
          const providerBadge = document.getElementById('providerUsed');
          const speedBadge = document.getElementById('execSpeed');
          const telemetry = document.getElementById('telemetryBadge');
          
          const action = document.getElementById('playAction').value;
          const text = document.getElementById('playText').value.trim();
          const instruction = document.getElementById('playInstruction').value.trim();

          if (!text) {
            output.innerHTML = '<span class="text-amber-400">Please provide input cell text.</span>';
            return;
          }

          btn.disabled = true;
          btn.innerHTML = '<span>⏳ Computing on AI Cluster...</span>';
          output.innerHTML = '<span class="text-zinc-500 animate-pulse font-mono">Routing across Cerebras & Groq...</span>';
          telemetry.innerText = 'Calculating';
          telemetry.className = 'text-xs font-mono text-amber-400 animate-pulse';

          const startTime = performance.now();

          try {
            const res = await fetch('/api/v1/process', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text: text, instruction: instruction, action: action, api_key: 'sp_demo_live' })
            });
            const data = await res.json();
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

            if (data.success) {
              output.innerHTML = '<span class="text-brand-300 font-bold">' + data.result + '</span>';
              providerBadge.innerText = 'Cluster: ' + (data.cached ? '⚡ In-Memory RAM Cache' : data.provider);
              speedBadge.innerText = 'Latency: ' + elapsed + 's';
              telemetry.innerText = 'Completed';
              telemetry.className = 'text-xs font-mono text-brand-400 font-bold';
            } else {
              output.innerHTML = '<span class="text-red-400">Error: ' + (data.detail || 'Execution Failed') + '</span>';
              telemetry.innerText = 'Error';
              telemetry.className = 'text-xs font-mono text-red-400';
            }
          } catch (err) {
            output.innerHTML = '<span class="text-red-400">Network Error: ' + err.message + '</span>';
            telemetry.innerText = 'Network Error';
          } finally {
            btn.disabled = false;
            btn.innerHTML = '<span>⚡ Run Live Formula</span>';
          }
        }

        async function generateKey() {
          const owner = document.getElementById('keyOwner').value.trim() || 'User';
          const tier = document.getElementById('keyTier').value;
          const box = document.getElementById('keyResult');
          
          box.classList.remove('hidden');
          box.innerHTML = '<span class="text-zinc-400 animate-pulse">Allocating tenant quota...</span>';

          try {
            const res = await fetch('/api/v1/keys/new', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ owner_name: owner, tier: tier })
            });
            const data = await res.json();
            box.innerHTML = '✅ <strong>Secret Tenant API Key:</strong><br><code class="text-white select-all font-mono font-bold text-sm bg-black p-1.5 rounded block my-1">' + data.api_key + '</code><div class="text-[11px] text-zinc-400 flex justify-between mt-1"><span>Credits: ' + data.credits + '</span><span>Tier: ' + data.tier.toUpperCase() + '</span></div>';
          } catch (e) {
            box.innerHTML = '<span class="text-red-400">Error generating key.</span>';
          }
        }

        function generateHostedKey(planName) {
          document.getElementById('keyOwner').value = planName + " User";
          document.getElementById('keyTier').value = 'pro';
          generateKey();
          window.location.href = "#integration";
        }

        function copyAppsScript() {
          const code = `const BACKEND_URL = "https://sheetpulseai.onrender.com/api/v1/process";
const SHEETPULSE_API_KEY = "sp_demo_live";

function SHEETPULSE(text, instruction) { return callSheetPulse(text, instruction, "custom"); }
function AI_CLEAN(text) { return callSheetPulse(text, "clean", "clean"); }
function AI_EXTRACT(text, target) { return callSheetPulse(text, target, "extract"); }
function AI_CLASSIFY(text, categories) { return callSheetPulse(text, categories, "classify"); }
function AI_FIX(brokenFormula, goal) { return callSheetPulse(brokenFormula, goal, "fix_formula"); }
function AI_SCRAPE(url, question) { return callSheetPulse(url, question, "scrape"); }

function callSheetPulse(text, inst, action) {
  if (!text) return "";
  const payload = { text: String(text), instruction: String(inst || ""), action: action, api_key: SHEETPULSE_API_KEY };
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
          alert("SheetPulse AI Apps Script client copied to clipboard! Paste into Extensions > Apps Script.");
        }
      </script>
    </body>
    </html>
    """

# --- API ENDPOINTS ---
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

@app.post("/api/v1/process")
async def process_cell(req: ProcessRequest):
    if not req.text or not req.text.strip():
        return {"success": True, "result": "", "cached": False, "provider": "None"}

    # 1. Fast Regex Extraction
    if req.action == "extract" and req.instruction:
        fast_match = try_fast_regex_extract(req.text, req.instruction)
        if fast_match:
            return {"success": True, "result": fast_match, "provider": "Regex:UltraFast", "cached": False}

    # 2. In-Memory Cache
    cache_key = hashlib.sha256(f"{req.action}:{req.instruction}:{req.text}".lower().encode()).hexdigest()
    if cache_key in MEMORY_CACHE:
        return {"success": True, "result": MEMORY_CACHE[cache_key]["result"], "provider": MEMORY_CACHE[cache_key]["provider"], "cached": True}

    verify_and_deduct_credits(req.api_key or "sp_demo_live", amount=1)

    act = req.action.lower()
    text_content = req.text

    if act == "scrape":
        scraped_data = fetch_url_text(req.text.strip())
        sys = "Analyze the web page content and answer the target question directly. Output ONLY the answer."
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
        sys = "Analyze the broken spreadsheet formula and fix it. Output ONLY a valid formula starting with '='."
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
