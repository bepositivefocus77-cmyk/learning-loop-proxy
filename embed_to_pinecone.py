"""
Embed Learning Loop Database into Pinecone (multi-user version)
--------------------------------------------------------------------
Reads interactions from learning_loop.db (SQLite) and upserts each into
Pinecone, partitioned by feedback status into namespaces:
    - "correct"    : responses confirmed correct
    - "incorrect"  : responses flagged wrong, with their correction stored as metadata
    - "pending"    : not yet reviewed

Every vector's metadata includes user_id, so retrieval in
learning_loop_proxy_auth.py can filter to only a given user's own
corrections -- no user ever sees another user's stored interactions.

Embeddings are generated locally using sentence-transformers, wrapped
behind get_embedding() so the embedding source can be swapped for a
hosted API later without touching the rest of this script.

Run with:
    python embed_to_pinecone.py
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

import db

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found. Check your .env file.")

INDEX_NAME = "learning-loop"
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2 output size

NAMESPACES = {
    "correct": "correct",
    "incorrect": "incorrect",
    None: "pending",
    "pending": "pending",
}

# ---------------------------------------------------------------------------
# Embedding function — swap this out later for a hosted embedding API
# without changing anything else in this script.
# ---------------------------------------------------------------------------
_local_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
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
# Main embedding loop -- now reads from SQLite across ALL users
# ---------------------------------------------------------------------------
def main():
    db.ensure_schema()
    conn = db.get_connection()
    rows = conn.execute("SELECT * FROM interactions").fetchall()
    conn.close()

    if not rows:
        print("No interactions found in the database. Nothing to embed.")
        return

    counts = {"correct": 0, "incorrect": 0, "pending": 0}

    for row in rows:
        entry = dict(row)

        interaction_id = entry["interaction_id"]
        user_id = entry["user_id"]
        prompt = entry.get("prompt", "")
        response = entry.get("response", "")
        feedback = entry.get("feedback")
        correction = entry.get("correction")
        provider = entry.get("provider")
        model_used = entry.get("model_used")

        namespace = NAMESPACES.get(feedback, "pending")
        vector = get_embedding(prompt)

        metadata = {
            "user_id": user_id,
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

    print("Done embedding database into Pinecone.")
    print(f"  correct:   {counts['correct']}")
    print(f"  incorrect: {counts['incorrect']}")
    print(f"  pending:   {counts['pending']}")
    print(f"  total interactions processed: {len(rows)}")


if __name__ == "__main__":
    main()