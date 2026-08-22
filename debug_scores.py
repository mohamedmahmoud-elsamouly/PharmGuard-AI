import chromadb
from src.rag import normalize_arabic, phrase_score, keyword_score, get_keywords
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

client = chromadb.PersistentClient('chroma_db')
col = client.get_collection('pharmabuddy_arabic')
data = col.get(where={'source': 'قواعد_خلط_الامبولات.txt'}, include=['documents','metadatas'])

question = "ما هي قواعد خلط الأمبولات؟"
keywords = get_keywords(question)
print(f"Keywords: {keywords}")

for d, m in zip(data['documents'], data['metadatas']):
    nd = normalize_arabic(d)
    ps = phrase_score(d, question)
    ks = keyword_score(d, keywords)
    contains_khalt = 'خلط الامبولات' in nd
    contains_khalt2 = 'خلط' in nd
    print(f"\nPage {m['page']}:")
    print(f"  phrase_score={ps} | keyword_score={ks}")
    print(f"  Has 'خلط الامبولات': {contains_khalt}")
    print(f"  Has 'خلط': {contains_khalt2}")
    print(f"  Normalized sample: {nd[:100]}")
