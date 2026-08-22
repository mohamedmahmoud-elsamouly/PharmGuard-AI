from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.rag import get_chroma_collection, normalize_arabic

collection = get_chroma_collection()

data = collection.get(
    include=["documents", "metadatas"]
)

documents = data["documents"]
metadatas = data["metadatas"]

terms = [
    "الأخطاء الدوائية",
    "خطأ دوائي",
    "الخطأ في إعطاء الدواء",
    "خطأ في إعطاء الدواء",
    "الحدث الدوائي الضائر",
]

print("=" * 80)
print("DIRECT TERM SEARCH")
print("=" * 80)

for term in terms:

    normalized_term = normalize_arabic(term)

    print(f"\n\nTERM: {term}")
    print("-" * 80)

    found = 0

    for document, metadata in zip(documents, metadatas):

        normalized_document = normalize_arabic(document)

        if normalized_term in normalized_document:

            found += 1

            print(
                f"\nSource: {metadata.get('source')}"
                f"\nPage: {metadata.get('page')}"
                f"\nChunk: {metadata.get('chunk_id')}"
            )

            print(document)

            print("-" * 80)

            if found >= 5:
                break

    print(f"FOUND: {found}")