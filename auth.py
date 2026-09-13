import hashlib
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from main import DBConn, is_byok_key

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

class SignupRequest(BaseModel):
    owner_name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=150)
    password: str = Field(..., min_length=6, max_length=100)

class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=150)
    password: str = Field(..., min_length=6, max_length=100)

def hash_password(password: str) -> str:
    salt = "sheetpulse_secure_salt_2026"
    return hashlib.sha256((password + salt).encode()).hexdigest()

@router.post("/signup")
def register_user(req: SignupRequest):
    clean_email = req.email.strip().lower()
    if "@" not in clean_email or "." not in clean_email:
        raise HTTPException(status_code=400, detail="Invalid email format.")

    hashed_pwd = hash_password(req.password)
    new_api_key = f"sp_{uuid.uuid4().hex[:18]}"
    created_time = time.time()

    with DBConn() as db:
        cur = db.execute("SELECT email FROM users WHERE email = ?", (clean_email,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Email is already registered. Please login.")

        db.execute(
            "INSERT INTO users (email, password_hash, owner_name, created_at) VALUES (?, ?, ?, ?)",
            (clean_email, hashed_pwd, req.owner_name.strip(), created_time)
        )

        db.execute(
            "INSERT INTO api_keys (key, owner_name, tier, credits_left, total_used, created_at) VALUES (?, ?, 'free', 100, 0, ?)",
            (new_api_key, req.owner_name.strip(), created_time)
        )

    return {
        "success": True,
        "message": "Account created successfully!",
        "api_key": new_api_key,
        "owner_name": req.owner_name.strip(),
        "email": clean_email,
        "credits": 100
    }

@router.post("/login")
def login_user(req: LoginRequest):
    clean_email = req.email.strip().lower()
    hashed_pwd = hash_password(req.password)

    with DBConn() as db:
        cur = db.execute("SELECT owner_name, password_hash FROM users WHERE email = ?", (clean_email,))
        user_row = cur.fetchone()

        if not user_row or user_row[1] != hashed_pwd:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        owner_name = user_row[0]

        cur_key = db.execute("SELECT key, tier, credits_left, total_used FROM api_keys WHERE owner_name = ? ORDER BY created_at DESC LIMIT 1", (owner_name,))
        key_row = cur_key.fetchone()

        if not key_row:
            new_api_key = f"sp_{uuid.uuid4().hex[:18]}"
            db.execute(
                "INSERT INTO api_keys (key, owner_name, tier, credits_left, total_used, created_at) VALUES (?, ?, 'free', 100, 0, ?)",
                (new_api_key, owner_name, time.time())
            )
            api_key, tier, credits_left, total_used = new_api_key, 'free', 100, 0
        else:
            api_key, tier, credits_left, total_used = key_row[0], key_row[1], key_row[2], key_row[3]

    return {
        "success": True,
        "message": "Login successful",
        "api_key": api_key,
        "owner_name": owner_name,
        "email": clean_email,
        "tier": tier.upper(),
        "credits_left": credits_left,
        "total_used": total_used
    }
