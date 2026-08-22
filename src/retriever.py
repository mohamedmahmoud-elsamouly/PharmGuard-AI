import re
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pharmabuddy_arabic"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# Arabic Normalization
# ============================================================

def normalize_arabic(text):

    if not text:
        return ""

    # إصلاح مشاكل استخراج PDF
    text = text.replace("\x9d", " ")
    text = text.replace("\uFFFD", " ")

    # إزالة التشكيل
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    # توحيد الهمزات
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ٱ", "ا")

    # توحيد التاء المربوطة
    text = text.replace("ة", "ه")

    # إزالة التطويل
    text = text.replace("ـ", "")

    # إزالة علامات الترقيم
    text = re.sub(
        r"[؟?!،؛:.,()\[\]{}\"'«»\-_/]",
        " ",
        text
    )

    # إزالة المسافات الزائدة
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Definition Question Detection
# ============================================================

def is_definition_question(question):

    normalized = normalize_arabic(question)

    patterns = [

        "ما هو",
        "ما هي",
        "ماهو",
        "ماهي",

        "ما معنى",
        "ماذا يعني",
        "ماذا تعني",

        "ما تعريف",
        "ما تعريفه",
        "ما تعريفها",

        "عرف",
        "تعريف",

    ]

    for pattern in patterns:

        pattern = normalize_arabic(pattern)

        if pattern in normalized:
            return True

    return False


# ============================================================
# Keyword Extraction
# ============================================================

def get_keywords(question):

    normalized = normalize_arabic(question)

    words = normalized.split()

    stop_words = {

        "ما",
        "هي",
        "هو",

        "من",
        "في",
        "عن",
        "على",
        "الى",

        "التي",
        "الذي",

        "هذا",
        "هذه",

        "هل",

        "ماهو",
        "ماهي",

        "ما",
        "ماذا",

        "اي",
        "أي",

        "معنى",
        "تعريف",

    }

    keywords = []

    for word in words:

        if word in stop_words:
            continue

        if len(word) < 3:
            continue

        keywords.append(word)

    return keywords


# ============================================================
# Keyword Score
# ============================================================

def keyword_score(document, keywords):

    normalized_document = normalize_arabic(
        document
    )

    score = 0

    for keyword in keywords:

        if keyword in normalized_document:

            score += 1

    return score


# ============================================================
# Phrase Score
# ============================================================

def phrase_score(document, question):

    document = normalize_arabic(
        document
    )

    question = normalize_arabic(
        question
    )

    score = 0

    # --------------------------------------------------------
    # Exact question
    # --------------------------------------------------------

    if question:

        if question in document:

            score += 20

    # --------------------------------------------------------
    # Important medical phrases
    # --------------------------------------------------------

    phrases = [

        "الاخطاء الدوائيه",
        "اخطاء دوائيه",

        "الاخطاء المرتبطه بالادويه",
        "اخطاء مرتبطه بالادويه",

        "الاخطاء المتعلقه بالادويه",
        "اخطاء متعلقه بالادويه",

        "خطا دوائي",
        "اخطاء الادويه",

        "سلامه الادويه",

        "الضرر المرتبط بالادويه",
        "الضرر المرتبط بالادويه",

        "خطا في اعطاء الدواء",

        "اخطاء في اعطاء الدواء",

    ]

    for phrase in phrases:

        phrase = normalize_arabic(
            phrase
        )

        if phrase in document:

            score += 5

    return min(score, 20)


# ============================================================
# Definition Score
# ============================================================

def definition_score(
    document,
    keywords,
    is_definition=False
):

    if not is_definition:

        return 0

    normalized = normalize_arabic(
        document
    )

    # --------------------------------------------------------
    # لازم يكون فيه على الأقل كلمة من السؤال
    # --------------------------------------------------------

    keyword_matches = 0

    for keyword in keywords:

        if keyword in normalized:

            keyword_matches += 1

    if keyword_matches == 0:

        return 0

    score = 0

    # --------------------------------------------------------
    # Strong definition patterns
    # --------------------------------------------------------

    strong_patterns = [

        "اي حدث",

        "اي اصابه",

        "عدم تنفيذ",

        "تطبيق خطه غير صحيحه",

        "ظرف او عامل او اجراء",

        "حدث لا ينبغي وقوعه",

        "حادثه لم تصب المريض",

    ]

    for pattern in strong_patterns:

        pattern = normalize_arabic(
            pattern
        )

        if pattern in normalized:

            score += 8

    # --------------------------------------------------------
    # General definition patterns
    # --------------------------------------------------------

    definition_patterns = [

        "التعريف",

        "يعرف",

        "يعني",

        "يقصد",

        "هو",

        "هي",

    ]

    for pattern in definition_patterns:

        pattern = normalize_arabic(
            pattern
        )

        if pattern in normalized:

            score += 2

    # --------------------------------------------------------
    # Direct medication-error definition
    # --------------------------------------------------------

    medical_definition_patterns = [

        "خطا في اعطاء الدواء",

        "استخدام غير مناسب للادويه",

        "الحاق الاذى بالمريض",

        "الدواء بيد اخصائي الرعايه الصحيه",

        "الدواء بيد المريض",

        "الدواء بيد المستهلك",

    ]

    for pattern in medical_definition_patterns:

        pattern = normalize_arabic(
            pattern
        )

        if pattern in normalized:

            score += 20

    # --------------------------------------------------------
    # Specific important phrase
    # --------------------------------------------------------

    if (
        "يمكن الوقايه منه"
        in normalized
    ):

        score += 15

    # --------------------------------------------------------
    # Keyword bonus
    # --------------------------------------------------------

    score += keyword_matches * 3

    # --------------------------------------------------------
    # Prevent score inflation
    # --------------------------------------------------------

    return min(score, 50)


# ============================================================
# Load All Chunks
# ============================================================

def load_all_chunks(collection):

    data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    chunks = []

    for document, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):

        try:

            chunk_id = int(
                metadata["chunk_id"]
            )

        except Exception:

            continue

        chunks.append({

            "document": document,

            "metadata": metadata,

            "chunk_id": chunk_id

        })

    return chunks


# ============================================================
# Context Expansion
# ============================================================

def expand_context(
    best_result,
    all_chunks,
    window=2
):

    source = best_result[
        "metadata"
    ]["source"]

    best_chunk_id = int(
        best_result[
            "metadata"
        ]["chunk_id"]
    )

    expanded = []

    for chunk in all_chunks:

        if (
            chunk["metadata"]["source"]
            != source
        ):

            continue

        distance = abs(
            chunk["chunk_id"]
            - best_chunk_id
        )

        if distance <= window:

            expanded.append(
                chunk
            )

    expanded.sort(
        key=lambda x:
        x["chunk_id"]
    )

    return expanded


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # ChromaDB
    # ========================================================

    print(
        "Connecting to ChromaDB..."
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Collection documents: "
        f"{collection.count()}"
    )

    # ========================================================
    # Load chunks
    # ========================================================

    print(
        "\nLoading all chunks..."
    )

    all_chunks = load_all_chunks(
        collection
    )

    print(
        f"Loaded chunks: "
        f"{len(all_chunks)}"
    )

    # ========================================================
    # Embedding Model
    # ========================================================

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    # ========================================================
    # Question
    # ========================================================

    question = input(
        "\nاكتب سؤالك بالعربي: "
    )

    if not question.strip():

        print(
            "من فضلك اكتب سؤالاً."
        )

        return

    # ========================================================
    # Keywords
    # ========================================================

    keywords = get_keywords(
        question
    )

    print(
        f"\nالكلمات المهمة: "
        f"{keywords}"
    )

    # ========================================================
    # Question Type
    # ========================================================

    definition_question = (
        is_definition_question(
            question
        )
    )

    print(
        f"Definition Question: "
        f"{definition_question}"
    )

    # ========================================================
    # Embedding
    # ========================================================

    print(
        "\nتحويل السؤال إلى Embedding..."
    )

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # ========================================================
    # Chroma Search
    # ========================================================

    print(
        "البحث داخل قاعدة المعرفة..."
    )

    results = collection.query(

        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=100

    )

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    distances = results[
        "distances"
    ][0]

    # ========================================================
    # Re-ranking
    # ========================================================

    scored = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        k_score = keyword_score(
            document,
            keywords
        )

        p_score = phrase_score(
            document,
            question
        )

        d_score = definition_score(
            document,
            keywords,
            definition_question
        )

        semantic_score = 1 / (
            1 + distance
        )

        # ----------------------------------------------------
        # Final Score
        # ----------------------------------------------------

        final_score = (

            # Phrase
            p_score

            # Keywords
            + (
                k_score
                * 1.5
            )

            # Definition
            + (
                d_score
                * 1.0
            )

            # Semantic
            + (
                semantic_score
                * 2
            )

        )

        scored.append({

            "document": document,

            "metadata": metadata,

            "distance": distance,

            "keyword_score": k_score,

            "phrase_score": p_score,

            "definition_score": d_score,

            "semantic_score": semantic_score,

            "final_score": final_score

        })

    # ========================================================
    # Sort
    # ========================================================

    scored.sort(

        key=lambda x:
        x["final_score"],

        reverse=True

    )

    # ========================================================
    # TOP 5
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "أفضل 5 نتائج بعد Re-ranking"
    )

    print(
        "=" * 70
    )

    for index, result in enumerate(
        scored[:5],
        start=1
    ):

        metadata = result[
            "metadata"
        ]

        print("\n")

        print(
            f"النتيجة #{index}"
        )

        print(
            "-" * 70
        )

        print(
            f"المصدر: "
            f"{metadata['source']}"
        )

        print(
            f"الصفحة: "
            f"{metadata['page']}"
        )

        print(
            f"Chunk ID: "
            f"{metadata['chunk_id']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"Phrase Score: "
            f"{result['phrase_score']}"
        )

        print(
            f"Keyword Score: "
            f"{result['keyword_score']}"
        )

        print(
            f"Definition Score: "
            f"{result['definition_score']}"
        )

        print(
            f"Semantic Score: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Final Score: "
            f"{result['final_score']:.4f}"
        )

        print(
            "\nTEXT:"
        )

        print(
            result["document"]
        )

    # ========================================================
    # BEST RESULT
    # ========================================================

    best = scored[0]

    metadata = best[
        "metadata"
    ]

    print("\n")

    print(
        "=" * 70
    )

    print(
        "BEST RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Source: "
        f"{metadata['source']}"
    )

    print(
        f"Page: "
        f"{metadata['page']}"
    )

    print(
        f"Chunk: "
        f"{metadata['chunk_id']}"
    )

    print(
        f"Final Score: "
        f"{best['final_score']:.4f}"
    )

    # ========================================================
    # Context Expansion
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "EXPANDED CONTEXT"
    )

    print(
        "=" * 70
    )

    expanded = expand_context(

        best,

        all_chunks,

        window=2

    )

    print(
        f"Number of chunks: "
        f"{len(expanded)}"
    )

    for chunk in expanded:

        metadata = chunk[
            "metadata"
        ]

        print("\n")

        print(
            f"--- Chunk "
            f"{metadata['chunk_id']} "
            f"| Page "
            f"{metadata['page']} ---"
        )

        print(
            chunk["document"]
        )

    # ========================================================
    # Finished
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "RETRIEVAL COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()