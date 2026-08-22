import os
import re
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import APIError

try:
    import streamlit as st
except ImportError:
    st = None


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pharmabuddy_arabic"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

GEMINI_MODEL = "gemini-3.6-flash"

ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


# ============================================================
# GEMINI
# ============================================================

def get_gemini_client():

    api_key = None

    if st is not None:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    api_key = str(api_key).strip()

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


# ============================================================
# ARABIC NORMALIZATION
# ============================================================

def normalize_arabic(text):

    if not text:
        return ""

    text = str(text)

    text = text.replace("\x9d", " ")
    text = text.replace("\ufffd", " ")

    # Arabic diacritics
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    # Alef
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
    )

    # Ya
    text = text.replace("ى", "ي")

    # Taa marbuta
    text = text.replace("ة", "ه")

    # Tatweel
    text = text.replace("ـ", "")

    # punctuation
    text = re.sub(
        r"[؟?!،؛:.,()\[\]{}\"'«»<>/\\|_\-–—]",
        " ",
        text
    )

    # whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# QUESTION TYPE
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
        "عرف",
        "تعريف",
        "المقصود",
    ]

    return any(
        normalize_arabic(pattern) in normalized
        for pattern in patterns
    )


# ============================================================
# MEDICAL TERM VARIANTS
# ============================================================

TERM_VARIANTS = {

    "الاخطاء الدوائيه": [
        "الاخطاء الدوائيه",
        "اخطاء دوائيه",
        "الخطا الدوائي",
        "خطا دوائي",
        "الاخطاء المرتبطه بالادويه",
        "اخطاء مرتبطه بالادويه",
        "الاخطاء المتعلقه بالادويه",
        "اخطاء متعلقه بالادويه",
    ],

    "خطا في اعطاء الدواء": [
        "خطا في اعطاء الدواء",
        "الخطا في اعطاء الدواء",
        "اخطاء في اعطاء الدواء",
        "الاخطاء في اعطاء الدواء",
    ],

    "حدث دوائي ضائر": [
        "حدث دوائي ضائر",
        "الحدث الدوائي الضائر",
        "الاحداث الدوائيه الضائره",
        "حدث ضائر",
    ],

    "سلامه المرضى": [
        "سلامه المرضى",
        "سلامه المريض",
    ],

    "مخاطر استخدام الادويه": [
        "مخاطر استخدام الادويه",
        "مخاطر الادويه",
        "مخاطر استخدام الدواء",
    ],

    "خلط الامبولات": [
        "خلط الامبولات",
        "خلط الحقن",
        "خلط الادويه",
        "خلط الادوية القابله للحقن",
        "قواعد خلط الامبولات",
        "قواعد خلط الحقن",
        "قواعد خلط الادويه",
        "شروط خلط الامبولات",
        "شروط خلط الحقن",
        "خلط الامبوله",
        "خلط امبولات",
        "توافق الحقن",
        "توافق الادويه",
        "توافق الادويه القابله للحقن",
        "عدم توافق الحقن",
        "عدم توافق الادويه",
        "الادويه التي تخلط",
        "الادويه التي لا تخلط",
        "خلط في السرنجه",
        "خلط في نفس السرنجه",
        "ادويه متوافقه",
        "ادويه غير متوافقه",
        "شروط خلط الحقن",
        "اهداف خلط الحقن",
    ],
}


# ============================================================
# QUERY EXPANSION
# ============================================================

def get_query_terms(question):

    normalized = normalize_arabic(question)

    terms = [normalized]

    # Add variants only when the concept exists
    for key, variants in TERM_VARIANTS.items():

        normalized_key = normalize_arabic(key)

        if normalized_key in normalized:

            terms.extend(
                normalize_arabic(v)
                for v in variants
            )

    # --------------------------------------------------------
    # Medication errors
    # --------------------------------------------------------

    if (
        "اخطاء دوائيه" in normalized
        or "خطا دوائي" in normalized
        or "الاخطاء الدوائيه" in normalized
    ):

        terms.extend(
            normalize_arabic(v)
            for v in TERM_VARIANTS[
                "الاخطاء الدوائيه"
            ]
        )

        terms.extend(
            normalize_arabic(v)
            for v in TERM_VARIANTS[
                "خطا في اعطاء الدواء"
            ]
        )

    # --------------------------------------------------------
    # Ampoules / injections
    # --------------------------------------------------------

    mixing_words = [
        "خلط",
        "امبول",
        "امبولا",
        "الامبول",
        "الامبوله",
        "حقن",
        "سرنجه",
        "توافق",
        "تخلط",
        "يخلط",
        "خلط الادويه",
    ]

    if any(
        word in normalized
        for word in mixing_words
    ):

        terms.extend(
            normalize_arabic(v)
            for v in TERM_VARIANTS[
                "خلط الامبولات"
            ]
        )

    return list(
        dict.fromkeys(
            t for t in terms if t
        )
    )


# ============================================================
# KEYWORDS
# ============================================================

def get_keywords(question):

    normalized = normalize_arabic(question)

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
        "ماذا",
        "اي",
        "أي",
        "معنى",
        "تعريف",
        "المقصود",
        "ما",
        "هو",
        "هي",
        "ماهي",
        "ماهو",
    }

    words = normalized.split()

    keywords = [
        word
        for word in words
        if (
            word not in stop_words
            and len(word) >= 2
        )
    ]

    return list(
        dict.fromkeys(keywords)
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    document,
    keywords,
):

    normalized_document = normalize_arabic(
        document
    )

    score = 0

    for keyword in keywords:

        if keyword in normalized_document:
            score += 1

    return score


# ============================================================
# QUERY TERM SCORE
# ============================================================

def query_term_score(
    document,
    question,
):

    normalized_document = normalize_arabic(
        document
    )

    terms = get_query_terms(question)

    score = 0

    for term in terms:

        if not term:
            continue

        if term in normalized_document:

            # Longer phrases are stronger
            words_count = len(term.split())

            if words_count >= 4:
                score += 12

            elif words_count >= 3:
                score += 8

            elif words_count >= 2:
                score += 5

            else:
                score += 2

    return min(score, 40)


# ============================================================
# EXACT PHRASE SCORE
# ============================================================

def phrase_score(
    document,
    question,
):

    normalized_document = normalize_arabic(
        document
    )

    normalized_question = normalize_arabic(
        question
    )

    score = 0

    # Exact question
    if (
        normalized_question
        and normalized_question in normalized_document
    ):
        score += 25

    # Important concepts
    important_phrases = [

        # Medication errors
        "الاخطاء الدوائيه",
        "اخطاء دوائيه",
        "الاخطاء المرتبطه بالادويه",
        "اخطاء مرتبطه بالادويه",
        "الاخطاء المتعلقه بالادويه",
        "اخطاء متعلقه بالادويه",
        "خطا دوائي",
        "الخطا الدوائي",
        "خطا في اعطاء الدواء",
        "الخطا في اعطاء الدواء",
        "اخطاء في اعطاء الدواء",

        # Adverse drug event
        "حدث دوائي ضائر",
        "الحدث الدوائي الضائر",

        # Patient safety
        "سلامه المرضى",
        "سلامه المريض",

        # Medication risks
        "مخاطر استخدام الادويه",
        "مخاطر الادويه",

        # Mixing
        "خلط الامبولات",
        "خلط الحقن",
        "خلط الادويه",
        "قواعد خلط الامبولات",
        "قواعد خلط الحقن",
        "قواعد خلط الادويه",
        "شروط خلط الامبولات",
        "شروط خلط الحقن",
        "توافق الحقن",
        "توافق الادويه",
        "عدم توافق الحقن",
        "عدم توافق الادويه",
        "خلط في السرنجه",
        "ادويه متوافقه",
        "ادويه غير متوافقه",
    ]

    for phrase in important_phrases:

        normalized_phrase = normalize_arabic(
            phrase
        )

        if normalized_phrase in normalized_document:

            # Give strong weight to real concept matches
            if len(normalized_phrase.split()) >= 3:
                score += 10
            else:
                score += 5

    return min(score, 50)


# ============================================================
# DEFINITION SCORE
# ============================================================

def definition_score(
    document,
    keywords,
    question,
):

    if not is_definition_question(question):
        return 0

    normalized = normalize_arabic(
        document
    )

    keyword_matches = sum(
        1
        for keyword in keywords
        if keyword in normalized
    )

    if keyword_matches == 0:
        return 0

    score = 0

    definition_patterns = [
        "التعريف",
        "يعرف",
        "يعني",
        "يقصد",
        "هو",
        "هي",
        "اي حدث",
        "اي اصابه",
        "عدم تنفيذ",
        "تطبيق خطه غير صحيحه",
        "ظرف او عامل او اجراء",
        "حدث لا ينبغي وقوعه",
    ]

    for pattern in definition_patterns:

        if normalize_arabic(pattern) in normalized:
            score += 3

    medication_definition_patterns = [
        "خطا في اعطاء الدواء",
        "استخدام غير مناسب للادويه",
        "الحاق الاذى بالمريض",
        "الدواء بيد اخصائي الرعايه الصحيه",
        "الدواء بيد المريض",
        "الدواء بيد المستهلك",
        "يمكن الوقايه منه",
    ]

    for pattern in medication_definition_patterns:

        if normalize_arabic(pattern) in normalized:
            score += 12

    if (
        "اخطاء دوائيه" in normalized
        or "خطا دوائي" in normalized
    ):
        score += 10

    score += min(
        keyword_matches * 3,
        12
    )

    return min(score, 50)


# ============================================================
# SEMANTIC SCORE
# ============================================================

def semantic_score(distance):

    try:

        distance = float(distance)

        # Chroma distance -> similarity-like score
        return 1.0 / (1.0 + distance)

    except Exception:

        return 0.0


# ============================================================
# CHROMA
# ============================================================

def get_chroma_collection():

    if not CHROMA_DIR.exists():

        raise FileNotFoundError(
            f"Chroma DB directory not found: {CHROMA_DIR}"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as error:

        raise ValueError(
            f"Chroma collection '{COLLECTION_NAME}' was not found."
        ) from error

    return collection


# ============================================================
# EMBEDDING MODEL
# ============================================================

_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return _embedding_model


# ============================================================
# LOAD ALL CHUNKS
# ============================================================

def load_all_chunks(
    collection
):

    data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    documents = data.get(
        "documents",
        []
    )

    metadatas = data.get(
        "metadatas",
        []
    )

    chunks = []

    for document, metadata in zip(
        documents,
        metadatas,
    ):

        try:

            chunk_id = int(
                metadata["chunk_id"]
            )

        except Exception:

            continue

        chunks.append(
            {
                "document": document,
                "metadata": metadata,
                "chunk_id": chunk_id,
            }
        )

    return chunks


# ============================================================
# CONTEXT EXPANSION
# ============================================================

def expand_context(
    best_result,
    all_chunks,
    window=1,
):

    source = best_result[
        "metadata"
    ].get("source")

    best_chunk_id = int(
        best_result[
            "metadata"
        ]["chunk_id"]
    )

    expanded = []

    for chunk in all_chunks:

        metadata = chunk["metadata"]

        if metadata.get("source") != source:
            continue

        try:

            chunk_id = int(
                metadata["chunk_id"]
            )

        except Exception:

            continue

        if abs(
            chunk_id - best_chunk_id
        ) <= window:

            expanded.append(chunk)

    expanded.sort(
        key=lambda x: int(
            x["metadata"]["chunk_id"]
        )
    )

    return expanded


# ============================================================
# SEARCH
# ============================================================

def search(
    question,
    limit=60,
):

    collection = get_chroma_collection()

    model = get_embedding_model()

    # --------------------------------------------------------
    # IMPORTANT
    # Use original Arabic question for embeddings.
    # --------------------------------------------------------

    query_embedding = model.encode(
        question,
        normalize_embeddings=False,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=limit,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    keywords = get_keywords(
        question
    )

    scored = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        k_score = keyword_score(
            document,
            keywords,
        )

        p_score = phrase_score(
            document,
            question,
        )

        q_score = query_term_score(
            document,
            question,
        )

        d_score = definition_score(
            document,
            keywords,
            question,
        )

        s_score = semantic_score(
            distance
        )

        # ====================================================
        # NEW BALANCED RANKING
        # ====================================================
        #
        # Semantic = keeps NEW questions working
        # Keyword  = supports direct matches
        # Phrase   = supports known concepts
        # Query    = supports expanded terminology
        # Definition = helps definition questions
        #
        # No single score dominates.
        # ====================================================

        final_score = (
            (s_score * 25.0)
            + (k_score * 2.0)
            + (p_score * 1.5)
            + (q_score * 1.5)
            + (d_score * 1.2)
        )

        scored.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "score": final_score,
                "keyword_score": k_score,
                "phrase_score": p_score,
                "query_term_score": q_score,
                "definition_score": d_score,
                "semantic_score": s_score,
            }
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    scored.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return scored


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    question,
    search_results,
    top_n=5,
):

    collection = get_chroma_collection()

    all_chunks = load_all_chunks(
        collection
    )

    selected = []

    selected_keys = set()

    # --------------------------------------------------------
    # Keep best evidence first
    # --------------------------------------------------------

    for result in search_results:

        if len(selected) >= top_n:
            break

        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "Unknown source"
        )

        try:

            chunk_id = int(
                metadata["chunk_id"]
            )

        except Exception:

            chunk_id = -1

        key = (
            source,
            chunk_id,
        )

        if key in selected_keys:
            continue

        selected.append(result)

        selected_keys.add(key)

    # --------------------------------------------------------
    # Context expansion
    #
    # Only expand around the strongest results.
    # --------------------------------------------------------

    strongest_results = selected[:3]

    expanded_chunks = []

    for result in strongest_results:

        expanded = expand_context(
            result,
            all_chunks,
            window=1,
        )

        for chunk in expanded:

            metadata = chunk["metadata"]

            source = metadata.get(
                "source",
                "Unknown"
            )

            try:

                chunk_id = int(
                    metadata["chunk_id"]
                )

            except Exception:

                continue

            key = (
                source,
                chunk_id,
            )

            if key in selected_keys:
                continue

            expanded_chunks.append(
                {
                    "document": chunk["document"],
                    "metadata": metadata,
                    "distance": 0,
                    "score": 0,
                    "keyword_score": 0,
                    "phrase_score": 0,
                    "query_term_score": 0,
                    "definition_score": 0,
                    "semantic_score": 0,
                }
            )

            selected_keys.add(key)

    # Add expanded chunks AFTER primary evidence
    selected.extend(
        expanded_chunks
    )

    return selected


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(
    context_chunks
):

    parts = []

    for index, chunk in enumerate(
        context_chunks,
        start=1,
    ):

        metadata = chunk["metadata"]

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown"
        )

        document = chunk["document"]

        parts.append(
            f"""
[Evidence {index}]
المصدر: {source}
الصفحة: {page}
Chunk: {chunk_id}

{document}
"""
        )

    return "\n\n".join(parts)


# ============================================================
# SOURCES
# ============================================================

def get_sources(
    context_chunks
):

    sources = []

    seen = set()

    for chunk in context_chunks:

        metadata = chunk["metadata"]

        source = metadata.get(
            "source"
        )

        page = metadata.get(
            "page"
        )

        if not source or page is None:
            continue

        key = (
            str(source),
            str(page),
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "page": page,
            }
        )

    return sources


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context_chunks,
):

    client = get_gemini_client()

    if client is None:

        return (
            "⚠️ لم يتم إعداد GEMINI_API_KEY. "
            "يرجى التحقق من إعدادات المفتاح."
        )

    if not context_chunks:

        return (
            "المعلومات المطلوبة غير موجودة "
            "في المصادر المتاحة."
        )

    context = format_context(
        context_chunks
    )

    definition_question = (
        is_definition_question(
            question
        )
    )

    prompt = f"""
أنت مساعد طبي عربي يعتمد فقط على
المصادر الطبية المسترجعة من نظام RAG.

السؤال:
{question}

هل السؤال يطلب تعريفًا؟
{"نعم" if definition_question else "لا"}

==================================================
المصادر المسترجعة
==================================================

{context}

==================================================
قواعد الإجابة
==================================================

1. أجب باللغة العربية.

2. استخدم المعلومات الموجودة في المصادر أعلاه فقط.

3. لا تستخدم أي معرفة خارج المصادر.

4. لا تخترع معلومات.

5. إذا كانت الإجابة موجودة بوضوح في أي دليل،
   أجب عنها مباشرة.

6. إذا وجدت أدلة متعددة متعلقة بالسؤال،
   اجمع المعلومات المتوافقة منها.

7. لا تجعل وجود معلومات غير كافية في دليل واحد
   سببًا لرفض الإجابة إذا كان دليل آخر يحتوي
   على الإجابة.

8. إذا كان السؤال عن "الأخطاء الدوائية"،
   استخدم الأدلة الخاصة بالأخطاء الدوائية.

9. إذا كان السؤال عن "خلط الأمبولات" أو
   "خلط الحقن" أو "توافق الأدوية القابلة للحقن"،
   استخدم أي دليل يتحدث عن هذه الموضوعات.

10. لا تعتبر "الحدث الدوائي الضائر" و"الخطأ الدوائي"
    مصطلحين مترادفين إلا إذا ذكر المصدر ذلك.

11. إذا كان السؤال يطلب تعريفًا،
    ابدأ بالتعريف مباشرة.

12. اذكر المصدر ورقم الصفحة عند الاستناد إلى المعلومة.

13. لا تذكر أسماء الـchunks.

14. لا تذكر تفاصيل ChromaDB أو embeddings.

15. لا تقل إنك AI.

16. اجعل الإجابة واضحة ومباشرة.

17. إذا كانت المعلومات المطلوبة غير موجودة فعلًا
    في جميع المصادر، اكتب:

"المعلومات المطلوبة غير موجودة في المصادر المتاحة."
"""

    max_attempts = 3

    for attempt in range(
        max_attempts
    ):

        try:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            text = getattr(
                response,
                "text",
                None
            )

            if text and text.strip():

                return text.strip()

            return (
                "المعلومات المطلوبة غير موجودة "
                "في المصادر المتاحة."
            )

        except APIError as error:

            error_text = str(error)

            lower_error = (
                error_text.lower()
            )

            if (
                "503" in error_text
                or "unavailable" in lower_error
                or "overloaded" in lower_error
                or "high demand" in lower_error
            ):

                if attempt < max_attempts - 1:

                    time.sleep(
                        2 ** attempt
                    )

                    continue

                return (
                    "⚠️ حدث ضغط مؤقت على خدمة Gemini. "
                    "حاول مرة أخرى بعد قليل."
                )

            if "404" in error_text:

                return (
                    "⚠️ نموذج Gemini المحدد غير متاح "
                    "لحساب API الحالي."
                )

            if (
                "401" in error_text
                or "403" in error_text
                or "permission" in lower_error
                or "api key" in lower_error
            ):

                return (
                    "⚠️ تعذر الوصول إلى خدمة Gemini. "
                    "تحقق من GEMINI_API_KEY."
                )

            return (
                "⚠️ حدث خطأ أثناء الاتصال بخدمة Gemini."
            )

        except Exception:

            if attempt < max_attempts - 1:

                time.sleep(
                    2 ** attempt
                )

                continue

            return (
                "⚠️ حدث خطأ غير متوقع أثناء "
                "توليد الإجابة."
            )


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def ask(
    question,
    search_limit=60,
    top_n=5,
):

    if not question or not question.strip():

        return {
            "answer": "يرجى كتابة سؤال.",
            "sources": [],
            "results": [],
        }

    question = question.strip()

    # Retrieval
    results = search(
        question,
        limit=search_limit,
    )

    if not results:

        return {
            "answer": (
                "المعلومات المطلوبة غير موجودة "
                "في المصادر المتاحة."
            ),
            "sources": [],
            "results": [],
        }

    # Context
    context_chunks = build_context(
        question,
        results,
        top_n=top_n,
    )

    # Gemini
    answer = generate_answer(
        question,
        context_chunks,
    )

    # Sources
    sources = get_sources(
        context_chunks
    )

    return {
        "answer": answer,
        "sources": sources,
        "results": results,
    }