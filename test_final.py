"""
Final retrieval verification script.
Tests:
1. New PDF: injection mixing questions
2. Old PDFs: medication safety questions
Run with: .venv\Scripts\python -X utf8 test_final.py
"""

import sys, warnings
warnings.filterwarnings('ignore')
from src.rag import search
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

SEPARATOR = "=" * 60

def run_query(label, query):
    print(f"\n{SEPARATOR}")
    print(f"TEST: {label}")
    print(f"QUERY: {query}")
    print(SEPARATOR)
    results = search(query, limit=40)
    top3 = results[:3]
    for i, r in enumerate(top3, 1):
        src = r['metadata'].get('source', '')
        page = r['metadata'].get('page', '')
        score = r['score']
        snippet = r['document'][:120].replace('\n', ' ')
        print(f"  [{i}] Source: {src} | Page: {page} | Score: {score:.2f}")
        print(f"      Text: {snippet}...")
    return top3

# ─── TEST 1: New PDF (injection mixing) ───────────────────────
r1 = run_query(
    "New PDF – injection mixing",
    "ما هي قواعد خلط الأمبولات؟"
)

# ─── TEST 2: New PDF (compatibility) ─────────────────────────
r2 = run_query(
    "New PDF – compatible drugs",
    "ما الأدوية التي يمكن خلطها مع بعضها؟"
)

# ─── TEST 3: New PDF (incompatibility) ───────────────────────
r3 = run_query(
    "New PDF – incompatible drugs",
    "ما الأدوية التي لا يجب خلطها مع بعضها؟"
)

# ─── TEST 4: Old PDF (medication safety) ─────────────────────
r4 = run_query(
    "Old PDF – medication safety",
    "ما هي الأخطاء الدوائية؟"
)

print(f"\n{SEPARATOR}")
print("SUMMARY")
print(SEPARATOR)

new_pdf_file = 'قواعد_خلط_الامبولات.txt'

def has_new_pdf(results):
    return any(r['metadata'].get('source') == new_pdf_file for r in results)

def has_old_pdf(results):
    return any(r['metadata'].get('source') != new_pdf_file for r in results)

checks = [
    ("Mixing rules query → new PDF retrieved", has_new_pdf(r1)),
    ("Compatible drugs query → new PDF retrieved", has_new_pdf(r2)),
    ("Incompatible drugs query → new PDF retrieved", has_new_pdf(r3)),
    ("Medication safety query → old PDF retrieved", has_old_pdf(r4)),
]

all_pass = True
for label, ok in checks:
    status = "✅ PASS" if ok else "❌ FAIL"
    if not ok:
        all_pass = False
    print(f"  {status}: {label}")

print()
print("OVERALL:", "✅ ALL PASSED" if all_pass else "❌ SOME TESTS FAILED")
