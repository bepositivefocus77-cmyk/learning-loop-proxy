"""
Learning Loop Proxy (multi-user, authenticated version)
----------------------------------------------------------
Adds email/password signup and login, per-user data isolation in SQLite,
a 30-messages-per-day usage cap per user, and per-user filtering of
Pinecone retrieval, on top of the same core capabilities as the original
single-user prototype (multi-provider chat, correction capture,
retrieval-augmented reuse, consent-boundary tagging).

Run with:
    uvicorn learning_loop_proxy_auth:app --reload
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from groq import Groq
from pinecone import Pinecone
from pydantic import BaseModel, EmailStr
from sentence_transformers import SentenceTransformer

import auth
import db

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash"

PINECONE_INDEX_NAME = "learning-loop"
pinecone_index = None
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing_indexes:
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)

_local_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    return _local_embedding_model.encode(text).tolist()


db.ensure_schema()

app = FastAPI(title="Learning Loop Proxy (Multi-user)")


# ---------------------------------------------------------------------------
# Consent boundary (same logic as the single-user version)
# ---------------------------------------------------------------------------
SENSITIVITY_LEVELS = ("public", "internal", "confidential", "unclassified")
EXTERNAL_PROVIDERS = {"groq", "gemini"}


def auto_detect_sensitivity(text: str) -> str | None:
    return None  # placeholder for future work


def check_sensitivity_policy(sensitivity: str, provider: str) -> str | None:
    if sensitivity == "confidential" and provider in EXTERNAL_PROVIDERS:
        return (
            f"This prompt was marked 'confidential' but was sent to an external "
            f"provider ('{provider}'). Review before reuse or further sharing."
        )
    return None


def resolve_sensitivity(explicit_sensitivity: str | None, prompt: str) -> str:
    if explicit_sensitivity:
        value = explicit_sensitivity.lower().strip()
        if value in SENSITIVITY_LEVELS:
            return value
    detected = auto_detect_sensitivity(prompt)
    return detected if detected in SENSITIVITY_LEVELS else "unclassified"


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header. Use 'Bearer <token>'.")

    token = authorization.removeprefix("Bearer ").strip()
    payload = auth.decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please log in again.")

    user = db.get_user_by_id(int(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return user


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChatRequest(BaseModel):
    prompt: str
    provider: str = "groq"
    sensitivity: str | None = None


class CorrectionRequest(BaseModel):
    interaction_id: str
    is_correct: bool
    correction: str | None = None


class ChatWithContextRequest(BaseModel):
    prompt: str
    provider: str = "groq"
    top_k: int = 3
    sensitivity: str | None = None


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/signup")
def signup(request: SignupRequest):
    if db.get_user_by_email(request.email):
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    if len(request.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    password_hash = auth.hash_password(request.password)
    user_id = db.create_user(request.email, password_hash)
    token = auth.create_access_token(user_id, request.email)
    return {"access_token": token, "token_type": "bearer", "user_id": user_id}


@app.post("/login")
def login(request: LoginRequest):
    user = db.get_user_by_email(request.email)
    if user is None or not auth.verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = auth.create_access_token(user["id"], user["email"])
    return {"access_token": token, "token_type": "bearer", "user_id": user["id"]}


# ---------------------------------------------------------------------------
# Core chat endpoints (all require auth)
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "running"}


@app.post("/chat")
def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    allowed, remaining, reset_at = db.check_and_consume_daily_quota(current_user["id"])
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Daily message limit ({db.DAILY_MESSAGE_LIMIT}) reached. Resets at {reset_at}.",
        )

    provider = request.provider.lower().strip()
    sensitivity = resolve_sensitivity(request.sensitivity, request.prompt)
    policy_warning = check_sensitivity_policy(sensitivity, provider)

    if provider == "groq":
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[{"role": "user", "content": request.prompt}],
            )
            answer = completion.choices[0].message.content
            model_used = GROQ_MODEL_NAME
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=400, detail="GEMINI_API_KEY not set in .env")
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            result = model.generate_content(request.prompt)
            answer = result.text
            model_used = GEMINI_MODEL_NAME
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider}'. Use 'groq' or 'gemini'.")

    interaction_id = db.insert_interaction(
        user_id=current_user["id"],
        prompt=request.prompt,
        response=answer,
        provider=provider,
        model_used=model_used,
        sensitivity=sensitivity,
        policy_warning=policy_warning,
    )

    return {
        "interaction_id": interaction_id,
        "response": answer,
        "provider": provider,
        "model_used": model_used,
        "sensitivity": sensitivity,
        "policy_warning": policy_warning,
        "messages_remaining_today": remaining,
    }


@app.post("/correct")
def correct(request: CorrectionRequest, current_user: dict = Depends(get_current_user)):
    existing = db.get_interaction(request.interaction_id, current_user["id"])
    if existing is None:
        raise HTTPException(status_code=404, detail="interaction_id not found for this account.")

    feedback = "correct" if request.is_correct else "incorrect"
    correction = request.correction if not request.is_correct else None
    db.update_feedback(request.interaction_id, current_user["id"], feedback, correction)

    updated = db.get_interaction(request.interaction_id, current_user["id"])
    return {"status": "updated", "entry": updated}


@app.get("/export")
def export_log(current_user: dict = Depends(get_current_user)):
    entries = db.get_all_interactions_for_user(current_user["id"])
    return {"entries": entries, "count": len(entries)}


def retrieve_context(prompt: str, top_k: int, user_id: int, min_score: float = 0.5) -> list[dict]:
    if pinecone_index is None:
        return []

    query_vector = get_embedding(prompt)
    matches = []

    for namespace in ("incorrect", "correct"):
        if len(matches) >= top_k:
            break
        remaining = top_k - len(matches)
        try:
            result = pinecone_index.query(
                vector=query_vector,
                top_k=remaining * 3,  # over-fetch since we filter by user_id after
                namespace=namespace,
                include_metadata=True,
                filter={"user_id": {"$eq": user_id}},
            )
        except Exception:
            continue
        for match in result.get("matches", [])[:remaining]:
            score = match.get("score", 0)
            if score < min_score:
                continue
            metadata = match.get("metadata", {})
            matches.append(
                {
                    "namespace": namespace,
                    "score": score,
                    "prompt": metadata.get("prompt"),
                    "response": metadata.get("response"),
                    "correction": metadata.get("correction"),
                }
            )

    return matches


def build_context_block(matches: list[dict]) -> str:
    if not matches:
        return ""
    lines = ["Relevant past interactions (for reference, do not repeat past mistakes):"]
    for m in matches:
        lines.append(f"- Past prompt: {m['prompt']}")
        if m["namespace"] == "incorrect" and m.get("correction"):
            lines.append(f"  Past response was WRONG. Correct answer: {m['correction']}")
        else:
            lines.append(f"  Past response (confirmed correct): {m['response']}")
    return "\n".join(lines)


@app.post("/chat-with-context")
def chat_with_context(request: ChatWithContextRequest, current_user: dict = Depends(get_current_user)):
    if pinecone_index is None:
        raise HTTPException(
            status_code=400,
            detail="Pinecone is not configured. Check PINECONE_API_KEY in .env and run embed_to_pinecone.py first.",
        )

    allowed, remaining, reset_at = db.check_and_consume_daily_quota(current_user["id"])
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Daily message limit ({db.DAILY_MESSAGE_LIMIT}) reached. Resets at {reset_at}.",
        )

    provider = request.provider.lower().strip()
    sensitivity = resolve_sensitivity(request.sensitivity, request.prompt)
    policy_warning = check_sensitivity_policy(sensitivity, provider)

    matches = retrieve_context(request.prompt, request.top_k, current_user["id"])
    context_block = build_context_block(matches)
    augmented_prompt = f"{context_block}\n\nNew question: {request.prompt}" if context_block else request.prompt

    if provider == "groq":
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[{"role": "user", "content": augmented_prompt}],
            )
            answer = completion.choices[0].message.content
            model_used = GROQ_MODEL_NAME
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=400, detail="GEMINI_API_KEY not set in .env")
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            result = model.generate_content(augmented_prompt)
            answer = result.text
            model_used = GEMINI_MODEL_NAME
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider}'. Use 'groq' or 'gemini'.")

    interaction_id = db.insert_interaction(
        user_id=current_user["id"],
        prompt=request.prompt,
        response=answer,
        provider=provider,
        model_used=model_used,
        sensitivity=sensitivity,
        policy_warning=policy_warning,
        used_context=[m["prompt"] for m in matches],
    )

    return {
        "interaction_id": interaction_id,
        "response": answer,
        "provider": provider,
        "model_used": model_used,
        "context_used": matches,
        "sensitivity": sensitivity,
        "policy_warning": policy_warning,
        "messages_remaining_today": remaining,
    }