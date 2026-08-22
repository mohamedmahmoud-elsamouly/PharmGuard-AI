from rag import get_chroma_collection, load_all_chunks, normalize_arabic


collection = get_chroma_collection()
chunks = load_all_chunks(collection)


search_terms = [
    "الأخطاء الدوائية",
    "اخطاء دوائية",
    "الخطأ الدوائي",
    "خطأ دوائي",
    "الأخطاء المرتبطة بالأدوية",
    "الأخطاء المتعلقة بالأدوية",
]


print("\n" + "=" * 80)
print("MEDICATION ERROR MATCHES")
print("=" * 80)


found = 0


for chunk in chunks:

    document = chunk["document"]
    normalized_document = normalize_arabic(document)

    matched_terms = []

    for term in search_terms:

        normalized_term = normalize_arabic(term)

        if normalized_term in normalized_document:
            matched_terms.append(term)

    if not matched_terms:
        continue

    found += 1

    metadata = chunk["metadata"]

    print(
        f"{found}. "
        f"Page={metadata.get('page')} | "
        f"Chunk={metadata.get('chunk_id')} | "
        f"Terms={matched_terms}"
    )


print("\n" + "=" * 80)
print(f"TOTAL MATCHES: {found}")
print("=" * 80)