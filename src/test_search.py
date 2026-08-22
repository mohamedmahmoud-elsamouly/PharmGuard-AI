from rag import search


question = "ما المقصود بالأخطاء الدوائية؟"

results = search(
    question,
    limit=60
)

print("\n" + "=" * 80)
print("SEARCH RESULTS")
print("=" * 80)

for i, result in enumerate(results[:10], start=1):

    metadata = result["metadata"]

    print("\n" + "-" * 80)

    print(f"RESULT: {i}")
    print(f"SOURCE: {metadata.get('source')}")
    print(f"PAGE: {metadata.get('page')}")
    print(f"CHUNK: {metadata.get('chunk_id')}")
    print(f"SCORE: {result.get('score')}")

    print("\nDOCUMENT:")
    print(result["document"][:1000])

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)