import chromadb
from sentence_transformers import SentenceTransformer
from src.rag import normalize_arabic, phrase_score, keyword_score, get_keywords, semantic_score
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(MODEL)
client = chromadb.PersistentClient('chroma_db')
col = client.get_collection('pharmabuddy_arabic')

question = "ما هي قواعد خلط الأمبولات؟"
keywords = get_keywords(question)

# Get all results sorted by vector distance
qe = model.encode(question).tolist()
res = col.query(query_embeddings=[qe], n_results=664)

docs = res['documents'][0]
metas = res['metadatas'][0]
dists = res['distances'][0]

target_src = 'قواعد_خلط_الامبولات.txt'

for i, (d, m, dist) in enumerate(zip(docs, metas, dists)):
    if m['source'] == target_src and m['page'] == '1':
        ss = semantic_score(dist)
        ps = phrase_score(d, question)
        ks = keyword_score(d, keywords)
        final = (ss * 8.0) + (ks * 2.0) + (ps * 2.5)
        print(f"FOUND at vector rank #{i+1}")
        print(f"Distance: {dist:.4f} | semantic_score: {ss:.4f}")
        print(f"phrase_score: {ps} | keyword_score: {ks}")
        print(f"FINAL SCORE: {final:.2f}")
        break
