import warnings
warnings.filterwarnings('ignore')
from src.rag import search
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

question = "ما هي قواعد خلط الأمبولات؟"
results = search(question, limit=40)

target = 'قواعد_خلط_الامبولات.txt'
print("TOP 5 SCORED RESULTS:")
for i, r in enumerate(results[:5], 1):
    src = r['metadata']['source']
    page = r['metadata']['page']
    print(f"  [{i}] {src} | Page {page} | Score {r['score']:.2f}")

print("\nRULES PDF ENTRIES:")
for i, r in enumerate(results, 1):
    if r['metadata']['source'] == target:
        print(f"  Rank #{i} | Page {r['metadata']['page']} | Score {r['score']:.2f}")
