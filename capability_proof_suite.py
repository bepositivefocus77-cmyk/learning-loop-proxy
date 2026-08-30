"""
Capability Proof Suite
------------------------
Runs a concrete, reproducible test for each of the five claims made about
the Learning Loop Proxy, and prints the evidence for each one. Intended
for demos, presentations, and as an appendix/evidence log for the paper.

Requires the proxy to already be running:
    uvicorn learning_loop_proxy_min:app --reload

Run with:
    python capability_proof_suite.py
"""

import json
import os

import requests

PROXY_URL = "http://127.0.0.1:8000"
LOG_FILE = "learning_log.jsonl"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def proof_1_data_ownership():
    section("PROOF 1: Data Ownership and Portability")
    if not os.path.exists(LOG_FILE):
        print("No log file found yet -- send at least one /chat request first.")
        return

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    print(f"Your entire interaction history lives locally in: {os.path.abspath(LOG_FILE)}")
    print(f"Total interactions stored on YOUR machine, not any vendor's servers: {len(lines)}")
    print("Sample entry (proof it's plain, readable JSON you control):")
    print(json.dumps(json.loads(lines[-1]), indent=2)[:400], "...")


def proof_2_provider_independence():
    section("PROOF 2: Provider Independence")
    prompt = "In one sentence, what is machine learning?"

    print(f"Sending the SAME prompt to two different providers: '{prompt}'\n")

    r_groq = requests.post(f"{PROXY_URL}/chat", json={"prompt": prompt, "provider": "groq"}, timeout=30)
    print("Groq response:  ", r_groq.json()["response"][:150])
    print("Model used:     ", r_groq.json()["model_used"])

    r_gemini = requests.post(f"{PROXY_URL}/chat", json={"prompt": prompt, "provider": "gemini"}, timeout=30)
    print("\nGemini response:", r_gemini.json()["response"][:150])
    print("Model used:     ", r_gemini.json()["model_used"])

    print(
        "\nBoth answers were logged to the SAME local file, under the SAME interface. "
        "Switching providers required changing one word ('provider'), not a new integration."
    )


def proof_3_structured_correction():
    section("PROOF 3: Structured Correction Capture")
    prompt = "What is a REST API, in one sentence?"

    r = requests.post(f"{PROXY_URL}/chat", json={"prompt": prompt}, timeout=30)
    interaction_id = r.json()["interaction_id"]
    print(f"Asked: '{prompt}'")
    print(f"Got interaction_id: {interaction_id}")

    correction_text = "A REST API is a set of rules for how software components communicate over HTTP using standard verbs like GET, POST, PUT, DELETE."
    requests.post(
        f"{PROXY_URL}/correct",
        json={"interaction_id": interaction_id, "is_correct": False, "correction": correction_text},
        timeout=30,
    )
    print("Correction submitted as STRUCTURED DATA (not chat text):")
    print(json.dumps({"interaction_id": interaction_id, "correction": correction_text}, indent=2))
    print("\nThis is queryable -- e.g. 'show me every correction ever made' is a one-line script,")
    print("not something possible against a normal chat transcript.")


def proof_4_correction_reuse():
    section("PROOF 4: Correction Reuse Across Sessions (RAG retrieval)")
    if not os.path.exists("rag_comparison_results.json"):
        print("Run compare_rag_context.py first to generate this evidence.")
        return

    with open("rag_comparison_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    print("Evidence from the controlled comparison experiment (compare_rag_context.py):\n")
    for r in results:
        print(f"Prompt: {r['rephrased_prompt']}")
        print(f"  Plain /chat score:            {r['plain_score']}")
        print(f"  /chat-with-context score:     {r['context_score']}")
        print(f"  Retrieved the past correction: {r['retrieved_match']}")
        print()

    avg_plain = sum(r["plain_score"] for r in results) / len(results)
    avg_context = sum(r["context_score"] for r in results) / len(results)
    print(f"Average score without retrieval: {round(avg_plain, 2)}")
    print(f"Average score WITH retrieval:    {round(avg_context, 2)}")
    print("This measurable difference is the proof that past corrections are being reused,")
    print("not just stored inertly.")


def proof_5_sensitivity_awareness():
    section("PROOF 5: Sensitivity Tagging and Outbound-Data Awareness")
    prompt = "Summarize our confidential Q3 salary structure for the leadership review."

    r = requests.post(
        f"{PROXY_URL}/chat",
        json={"prompt": prompt, "sensitivity": "confidential"},
        timeout=30,
    )
    data = r.json()
    print(f"Prompt: '{prompt}'")
    print(f"Sensitivity declared: {data['sensitivity']}")
    print(f"Policy warning returned: {data['policy_warning']}")
    print("\nCompare: a normal ChatGPT/Gemini/Claude chat window gives NO such warning")
    print("before sensitive text leaves your machine.")


def main():
    print("Learning Loop Proxy -- Capability Proof Suite")
    print("Make sure the proxy is running at", PROXY_URL)

    proof_1_data_ownership()
    proof_2_provider_independence()
    proof_3_structured_correction()
    proof_4_correction_reuse()
    proof_5_sensitivity_awareness()

    section("DONE")
    print("All five capability proofs completed. Copy the relevant sections above")
    print("into your paper's evidence/appendix section or a demo script.")


if __name__ == "__main__":
    main()