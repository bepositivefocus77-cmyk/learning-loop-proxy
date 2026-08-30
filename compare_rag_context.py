"""
RAG Context Comparison Experiment
------------------------------------
For each stored correction (an entry in the 'incorrect' namespace with a
real correction text), this script asks a REPHRASED version of that
original question through both:

    /chat               (no retrieval -- plain LLM answer)
    /chat-with-context  (retrieval-augmented -- past correction injected)

It then checks whether the stored correction's key content actually shows
up in the context-augmented answer but not the plain one, and prints a
clean before/after table suitable for a paper's evaluation section.

This does NOT require the OpenAI/Groq/Gemini keys directly -- it just
calls your already-running proxy's HTTP endpoints.

Run with (proxy must already be running):
    python compare_rag_context.py
"""

import json
import os

import requests

PROXY_URL = "http://127.0.0.1:8000"
LOG_FILE = "learning_log.jsonl"

# Rephrased versions of questions where we have a stored correction.
# Map: interaction_id of the ORIGINAL incorrect entry -> a rephrased prompt
# that a real user might ask instead of repeating the exact original wording.
REPHRASED_PROMPTS = {
    "edb6cc96-2fc5-454b-97b8-ac01ba5add6f": "Can you give me a solid intro to FastAPI as a framework?",
    "aca8e978-ef37-4374-b9bf-5a403e7aefe2": "How do I protect my LLM app from prompt injection attacks?",
    "96968862-1767-4530-a2d2-72c33a90c551": "Should I fine-tune a model or just use RAG for my use case?",
}


def load_corrections():
    """Load the original prompt + correction for each incorrect entry we have a rephrase for."""
    corrections = {}
    if not os.path.exists(LOG_FILE):
        return corrections

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            iid = entry.get("interaction_id")
            if iid in REPHRASED_PROMPTS and entry.get("correction"):
                corrections[iid] = {
                    "original_prompt": entry.get("prompt"),
                    "correction": entry.get("correction"),
                }
    return corrections


def call_plain_chat(prompt):
    r = requests.post(f"{PROXY_URL}/chat", json={"prompt": prompt}, timeout=60)
    r.raise_for_status()
    return r.json()["response"]


def call_context_chat(prompt):
    r = requests.post(f"{PROXY_URL}/chat-with-context", json={"prompt": prompt}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["response"], data.get("context_used", [])


def keyword_overlap_score(text, correction):
    """
    Rough, transparent proxy metric: what fraction of the correction's
    distinctive words (length > 4, to skip filler words) appear in the
    answer? Not a rigorous NLP metric, but simple, reproducible, and
    honestly reported as such in the paper.
    """
    correction_words = {w.lower().strip(".,:;") for w in correction.split() if len(w) > 4}
    if not correction_words:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for w in correction_words if w in text_lower)
    return round(matched / len(correction_words), 2)


def main():
    corrections = load_corrections()
    if not corrections:
        print("No matching corrections found for the configured REPHRASED_PROMPTS. Check interaction IDs.")
        return

    results = []

    for iid, info in corrections.items():
        rephrased = REPHRASED_PROMPTS[iid]
        print(f"\n=== Original correction ({iid}) ===")
        print("Original prompt:", info["original_prompt"])
        print("Stored correction:", info["correction"][:150], "...")
        print("Rephrased test prompt:", rephrased)

        plain_answer = call_plain_chat(rephrased)
        context_answer, context_used = call_context_chat(rephrased)

        plain_score = keyword_overlap_score(plain_answer, info["correction"])
        context_score = keyword_overlap_score(context_answer, info["correction"])
        retrieved_the_right_one = any(
            m.get("prompt") == info["original_prompt"] for m in context_used
        )

        print(f"Plain /chat overlap score:            {plain_score}")
        print(f"/chat-with-context overlap score:     {context_score}")
        print(f"Retrieved the matching correction:    {retrieved_the_right_one}")

        results.append(
            {
                "interaction_id": iid,
                "rephrased_prompt": rephrased,
                "plain_score": plain_score,
                "context_score": context_score,
                "retrieved_match": retrieved_the_right_one,
                "plain_answer": plain_answer,
                "context_answer": context_answer,
            }
        )

    print("\n\n=== SUMMARY TABLE (paste into paper) ===")
    print(f"{'Prompt':<45} {'Plain':<8} {'W/ Context':<12} {'Retrieved?':<10}")
    for r in results:
        print(f"{r['rephrased_prompt'][:43]:<45} {r['plain_score']:<8} {r['context_score']:<12} {str(r['retrieved_match']):<10}")

    avg_plain = sum(r["plain_score"] for r in results) / len(results)
    avg_context = sum(r["context_score"] for r in results) / len(results)
    print(f"\nAverage overlap score -- plain /chat: {round(avg_plain, 2)}")
    print(f"Average overlap score -- /chat-with-context: {round(avg_context, 2)}")

    with open("rag_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nFull results (including full answer text) saved to rag_comparison_results.json")


if __name__ == "__main__":
    main()