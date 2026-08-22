import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pharmabuddy_arabic"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

model = SentenceTransformer(
    MODEL_NAME
)


question = "ما هي الأخطاء الدوائية؟"

target_id = "WHO_Global_Patient_Safety_AR.txt_101_366"


# Create embeddings
question_embedding = model.encode(
    question,
    normalize_embeddings=True
)

result = collection.get(
    ids=[target_id],
    include=["documents"]
)

document = result["documents"][0]

document_embedding = model.encode(
    document,
    normalize_embeddings=True
)


similarity = np.dot(
    question_embedding,
    document_embedding
)


print("=" * 70)

print("Question:")
print(question)

print("\nTarget Chunk:")
print(document)

print("\nCosine Similarity:")

print(similarity)

print("=" * 70)