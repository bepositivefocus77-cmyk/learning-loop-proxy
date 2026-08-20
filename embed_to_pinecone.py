"""
Embed Learning Log into Pinecone
----------------------------------
Reads learning_log.jsonl and upserts each entry into Pinecone,
partitioned by feedback status into separate namespaces:
    - "correct"    : responses confirmed correct
    - "incorrect"  : responses flagged wrong, with their correction stored as metadata
    - "pending"    : not yet reviewed

Embeddings are generated locally using sentence-transformers, wrapped
behind get_embedding() so the embedding source can be swapped for a
hosted API later without touching the rest of this script.

Run with:
    python embed_to_pinecone.py
"""

import json
import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found. Check your .env file.")

LOG_FILE = "learning_log.jsonl"
INDEX_NAME = "learning-loop"
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2 output size

NAMESPACES = {
    "correct": "correct",
    "incorrect": "incorrect",
    None: "pending",       # feedback field is null -> pending
    "pending": "pending",
}

# ---------------------------------------------------------------------------
# Embedding function — swap this out later for a hosted embedding API
# without changing anything else in this script.
# ---------------------------------------------------------------------------
_local_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """Return a vector embedding for the given text.

    Currently backed by a local sentence-transformers model.
    To switch to a hosted embedding API later, replace the body of this
    function only -- callers don't need to change.
    """
    return _local_model.encode(text).tolist()


# ---------------------------------------------------------------------------
# Pinecone setup
# ---------------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)


# ---------------------------------------------------------------------------
# Main embedding loop
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(LOG_FILE):
        print(f"No log file found at {LOG_FILE}. Nothing to embed.")
        return

    counts = {"correct": 0, "incorrect": 0, "pending": 0, "skipped": 0}

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            interaction_id = entry.get("interaction_id")
            if not interaction_id:
                # Entries from before interaction_id existed can't be
                # reliably upserted/updated later, so we skip them.
                counts["skipped"] += 1
                continue

            prompt = entry.get("prompt", "")
            response = entry.get("response", "")
            feedback = entry.get("feedback")
            correction = entry.get("correction")
            provider = entry.get("provider")
            model_used = entry.get("model_used")

            namespace = NAMESPACES.get(feedback, "pending")

            vector = get_embedding(prompt)

            metadata = {
                "prompt": prompt,
                "response": response,
                "provider": provider or "unknown",
                "model_used": model_used or "unknown",
                "feedback": feedback or "pending",
            }
            if correction:
                metadata["correction"] = correction

            index.upsert(
                vectors=[{"id": interaction_id, "values": vector, "metadata": metadata}],
                namespace=namespace,
            )

            counts[namespace] += 1

    print("Done embedding log into Pinecone.")
    print(f"  correct:   {counts['correct']}")
    print(f"  incorrect: {counts['incorrect']}")
    print(f"  pending:   {counts['pending']}")
    print(f"  skipped (no interaction_id): {counts['skipped']}")


if __name__ == "__main__":
    main()