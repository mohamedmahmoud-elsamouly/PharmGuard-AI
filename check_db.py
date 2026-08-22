import chromadb
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
client = chromadb.PersistentClient('chroma_db')
collection = client.get_collection('pharmabuddy_arabic')
data = collection.get(include=["metadatas"])
sources = set([m['source'] for m in data['metadatas']])
print(sources)
data_q = collection.get(where={'source': 'قواعد_خلط_الامبولات.txt'})
print(f"Chunks for rules PDF: {len(data_q['ids'])}")
