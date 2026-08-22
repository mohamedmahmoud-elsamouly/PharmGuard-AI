from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_DIR = Path("data/chunks")
CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "pharmabuddy_arabic"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks():

    chunks = []

    chunk_files = CHUNKS_DIR.glob("*_chunks.txt")

    for file_path in chunk_files:

        content = file_path.read_text(
            encoding="utf-8"
        )

        raw_chunks = content.split("=" * 80)

        for raw_chunk in raw_chunks:

            raw_chunk = raw_chunk.strip()

            if not raw_chunk:
                continue

            lines = raw_chunk.splitlines()

            chunk_id = None
            source = None
            page = None
            text_start = None

            for i, line in enumerate(lines):

                if line.startswith("Chunk ID:"):
                    chunk_id = line.replace(
                        "Chunk ID:", ""
                    ).strip()

                elif line.startswith("Source:"):
                    source = line.replace(
                        "Source:", ""
                    ).strip()

                elif line.startswith("Page:"):
                    page = line.replace(
                        "Page:", ""
                    ).strip()

                elif line.startswith("-" * 10):
                    text_start = i + 1
                    break

            if text_start is None:
                continue

            text = "\n".join(
                lines[text_start:]
            ).strip()

            if not text:
                continue

            chunks.append(
                {
                    "id": f"{source}_{page}_{chunk_id}",
                    "text": text,
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_id
                }
            )

    return chunks


def main():

    print("Loading chunks...")

    chunks = load_chunks()

    print(
        f"Loaded chunks: {len(chunks)}"
    )

    if not chunks:

        print("No chunks found.")

        return

    print(
        "\nLoading multilingual embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    print("\nCreating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    )

    print(
        "Embeddings created successfully!"
    )

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        "\nAdding documents to ChromaDB..."
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("\n" + "=" * 60)

    print(
        "PharmaBuddy Arabic Knowledge Base Ready!"
    )

    print("=" * 60)

    print(
        f"Total documents: {collection.count()}"
    )


if __name__ == "__main__":
    main()