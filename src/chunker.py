from pathlib import Path
import re


PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_text(text):

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) <= CHUNK_SIZE:

            current_chunk += paragraph + "\n"

        else:

            if current_chunk.strip():
                chunks.append(
                    current_chunk.strip()
                )

            overlap = current_chunk[-CHUNK_OVERLAP:]

            if " " in overlap:
                overlap = overlap[
                    overlap.find(" ") + 1:
                ]

            current_chunk = (
                overlap + "\n" + paragraph + "\n"
            )

    if current_chunk.strip():
        chunks.append(
            current_chunk.strip()
        )

    return chunks


def extract_page_sections(text):

    pattern = r"--- Page (\d+) ---"

    matches = list(
        re.finditer(pattern, text)
    )

    sections = []

    for i, match in enumerate(matches):

        page_number = int(
            match.group(1)
        )

        start = match.end()

        if i + 1 < len(matches):

            end = matches[i + 1].start()

        else:

            end = len(text)

        page_text = text[start:end].strip()

        if page_text:

            sections.append(
                {
                    "page": page_number,
                    "text": page_text
                }
            )

    return sections


def process_file(file_path):

    text = file_path.read_text(
        encoding="utf-8"
    )

    pages = extract_page_sections(
        text
    )

    all_chunks = []

    for page in pages:

        page_chunks = split_text(
            page["text"]
        )

        for chunk in page_chunks:

            all_chunks.append(
                {
                    "page": page["page"],
                    "text": chunk
                }
            )

    output_file = (
        CHUNKS_DIR /
        f"{file_path.stem}_chunks.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for i, chunk in enumerate(
            all_chunks
        ):

            file.write(
                "=" * 80
            )

            file.write("\n")

            file.write(
                f"Chunk ID: {i}\n"
            )

            file.write(
                f"Source: {file_path.name}\n"
            )

            file.write(
                f"Page: {chunk['page']}\n"
            )

            file.write(
                "-" * 80
            )

            file.write("\n")

            file.write(
                chunk["text"]
            )

            file.write("\n\n")

    return len(all_chunks)


def main():

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    text_files = list(
        PROCESSED_DIR.glob("*.txt")
    )

    if not text_files:

        print(
            "No processed text files found."
        )

        return

    total_chunks = 0

    for file_path in text_files:

        print(
            f"\nProcessing: {file_path.name}"
        )

        count = process_file(
            file_path
        )

        print(
            f"Created chunks: {count}"
        )

        total_chunks += count

    print("\n" + "=" * 60)

    print(
        f"Total chunks: {total_chunks}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()