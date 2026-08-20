"""
Learning Loop Proxy (minimal version)
--------------------------------------
Forwards prompts to Groq or Gemini and logs each prompt + response locally,
so you retain your own interaction history instead of it living
only inside any single AI provider's system. Switching providers doesn't
lose your accumulated log -- that's the point.

Run with:
    uvicorn learning_loop_proxy_min:app --reload
"""

import json
import os
import uuid
from datetime import datetime, timezone

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pinecone import Pinecone
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load .env from this script's own folder (fixes uvicorn --reload subprocess issues)
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

LOG_FILE = "learning_log.jsonl"

# ---------------------------------------------------------------------------
# Pinecone + embedding setup (for /chat-with-context)
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME = "learning-loop"
pinecone_index = None
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing_indexes:
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)

_local_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """Return a vector embedding for the given text.

    Currently backed by a local sentence-transformers model. To switch to
    a hosted embedding API later, replace the body of this function only.
    """
    return _local_embedding_model.encode(text).tolist()


app = FastAPI(title="Learning Loop Proxy (Minimal)")


# ---------------------------------------------------------------------------
# Consent Boundary — data-sensitivity tagging and policy layer
# ---------------------------------------------------------------------------
# Every provider currently supported (Groq, Gemini) is an external, third-party
# API. This layer does not block calls to them -- it classifies each prompt's
# sensitivity and, when a confidential prompt is about to leave the local
# environment, attaches a clear warning so the caller can make an informed
# decision. This is a "consent boundary" in the sense Nadella describes:
# visibility and warning before data crosses the boundary, not a hard block.

SENSITIVITY_LEVELS = ("public", "internal", "confidential", "unclassified")
EXTERNAL_PROVIDERS = {"groq", "gemini"}  # both leave the local environment today


def auto_detect_sensitivity(text: str) -> str | None:
    """
    Placeholder for future automatic sensitivity classification.

    Not implemented yet -- always returns None, which causes the caller
    to fall back to "unclassified". Intended future implementations:
    keyword/regex matching against known-sensitive terms, or a small
    local classifier model. Keeping this as an isolated function now
    means auto-detection can be dropped in later without changing any
    call site in /chat or /chat-with-context.
    """
    return None


def check_sensitivity_policy(sensitivity: str, provider: str) -> str | None:
    """
    Evaluate whether sending data of this sensitivity to this provider
    should raise a warning. Returns a warning string if so, else None.

    Current policy: confidential data sent to any external provider is
    allowed to proceed, but flagged with an explicit warning that is
    both returned to the caller and recorded in the log, so there is
    an auditable trail of every boundary crossing.
    """
    if sensitivity == "confidential" and provider in EXTERNAL_PROVIDERS:
        return (
            f"This prompt was marked 'confidential' but was sent to an external "
            f"provider ('{provider}'). Review before reuse or further sharing."
        )
    return None


def resolve_sensitivity(explicit_sensitivity: str | None, prompt: str) -> str:
    """
    Determine the sensitivity level to use for a request: the caller's
    explicit value takes priority; if not provided, fall back to
    auto-detection (currently a stub); if that also yields nothing,
    default to "unclassified".
    """
    if explicit_sensitivity:
        value = explicit_sensitivity.lower().strip()
        if value in SENSITIVITY_LEVELS:
            return value
    detected = auto_detect_sensitivity(prompt)
    return detected if detected in SENSITIVITY_LEVELS else "unclassified"


class ChatRequest(BaseModel):
    prompt: str
    provider: str = "groq"  # "groq" or "gemini"
    sensitivity: str | None = None  # "public" | "internal" | "confidential"; defaults to auto-detect -> "unclassified"


class CorrectionRequest(BaseModel):
    interaction_id: str
    is_correct: bool
    correction: str | None = None  # only needed if is_correct is False


class ChatWithContextRequest(BaseModel):
    prompt: str
    provider: str = "groq"  # "groq" or "gemini"
    top_k: int = 3  # how many past interactions to retrieve as context
    sensitivity: str | None = None  # "public" | "internal" | "confidential"; defaults to auto-detect -> "unclassified"


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/chat")
def chat(request: ChatRequest):
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

    # Log the interaction
    interaction_id = str(uuid.uuid4())
    entry = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": request.prompt,
        "response": answer,
        "provider": provider,
        "model_used": model_used,
        "feedback": None,
        "correction": None,
        "sensitivity": sensitivity,
        "policy_warning": policy_warning,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "interaction_id": interaction_id,
        "response": answer,
        "provider": provider,
        "model_used": model_used,
        "sensitivity": sensitivity,
        "policy_warning": policy_warning,
    }


@app.post("/correct")
def correct(request: CorrectionRequest):
    """
    Attach feedback (and an optional correction) to a previously logged
    interaction. This captures the most valuable data point: where the
    model was wrong and what the right answer should have been.
    """
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="No log file found yet.")

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_entry = None
    for i, line in enumerate(lines):
        entry = json.loads(line)
        if entry.get("interaction_id") == request.interaction_id:
            entry["feedback"] = "correct" if request.is_correct else "incorrect"
            entry["correction"] = request.correction if not request.is_correct else None
            lines[i] = json.dumps(entry, ensure_ascii=False) + "\n"
            updated_entry = entry
            break

    if updated_entry is None:
        raise HTTPException(status_code=404, detail="interaction_id not found in log")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return {"status": "updated", "entry": updated_entry}


@app.get("/export")
def export_log():
    """
    Return the full learning log so you can own/export your accumulated
    interaction + correction history at any time, independent of any
    single provider.
    """
    if not os.path.exists(LOG_FILE):
        return {"entries": [], "count": 0}

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    return {"entries": entries, "count": len(entries)}


def retrieve_context(prompt: str, top_k: int, min_score: float = 0.5) -> list[dict]:
    """
    Search Pinecone for past interactions relevant to this prompt.
    Prioritizes the 'incorrect' namespace first (corrections are the most
    valuable signal -- "don't repeat this mistake"), then fills any
    remaining slots from the 'correct' namespace.

    Only matches with cosine similarity >= min_score are kept, so
    unrelated prompts don't get irrelevant context forced in just
    because Pinecone always returns its closest available vectors.
    """
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
                top_k=remaining,
                namespace=namespace,
                include_metadata=True,
            )
        except Exception:
            continue
        for match in result.get("matches", []):
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
    """Turn retrieved past interactions into a short context block for the prompt."""
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
def chat_with_context(request: ChatWithContextRequest):
    """
    Same as /chat, but first retrieves relevant past interactions and
    corrections from Pinecone and injects them as context before calling
    the LLM. This is the retrieval-augmented version of the proxy: it
    lets past human corrections actually influence future answers,
    independent of which provider is used.
    """
    if pinecone_index is None:
        raise HTTPException(
            status_code=400,
            detail="Pinecone is not configured. Check PINECONE_API_KEY in .env and run embed_to_pinecone.py first.",
        )

    provider = request.provider.lower().strip()
    sensitivity = resolve_sensitivity(request.sensitivity, request.prompt)
    policy_warning = check_sensitivity_policy(sensitivity, provider)

    matches = retrieve_context(request.prompt, request.top_k)
    context_block = build_context_block(matches)

    if context_block:
        augmented_prompt = f"{context_block}\n\nNew question: {request.prompt}"
    else:
        augmented_prompt = request.prompt

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

    interaction_id = str(uuid.uuid4())
    entry = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": request.prompt,
        "response": answer,
        "provider": provider,
        "model_used": model_used,
        "feedback": None,
        "correction": None,
        "used_context": [m["prompt"] for m in matches],
        "sensitivity": sensitivity,
        "policy_warning": policy_warning,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "interaction_id": interaction_id,
        "response": answer,
        "provider": provider,
        "model_used": model_used,
        "context_used": matches,
        "sensitivity": sensitivity,
        "policy_warning": policy_warning,
    }