import streamlit as st
import sys
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ============================================================
# IMPORT RAG
# ============================================================

from rag import ask


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PharmGuard AI",
    page_icon="💊",
    layout="centered"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💊 PharmGuard AI")

    st.caption(
        "مساعد الصيدلي الذكي لفحص التعارضات الدوائية وضمان أمان صرف العلاجات."
    )

    st.markdown("---")

    if st.button(
        "🗑️ مسح المحادثة",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 style="
        text-align:center;
        margin-bottom:5px;
    ">
        💊 PharmGuard AI
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        text-align:center;
        color:gray;
        font-size:16px;
    ">
        مساعد الصيدلي الذكي لفحص التعارضات الدوائية وضمان أمان صرف العلاجات
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div style="
            text-align:center;
            color:gray;
            padding:20px;
        ">
            👋 أهلاً بك في PharmGuard AI<br>
            اكتب سؤالك الطبي باللغة العربية.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

        if (
            msg["role"] == "assistant"
            and msg.get("sources")
        ):

            st.markdown(
                "### 📚 المصادر"
            )

            for source in msg["sources"]:

                st.markdown(
                    f"- `{source}`"
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "اكتب سؤالك الطبي بالعربي هنا..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "جاري البحث في المصادر الطبية وتوليد الإجابة..."
        ):

            try:

                result = ask(
                    question,
                    search_limit=60,
                    top_n=5
                )

                answer_text = result[
                    "answer"
                ]

                sources_list = []

                # ------------------------------------------------
                # Sources
                # ------------------------------------------------

                for source_info in result[
                    "sources"
                ]:

                    source = source_info[
                        "source"
                    ]

                    page = source_info[
                        "page"
                    ]

                    source_text = (
                        f"{source} — صفحة {page}"
                    )

                    if source_text not in sources_list:

                        sources_list.append(
                            source_text
                        )

                # ------------------------------------------------
                # Display answer
                # ------------------------------------------------

                st.markdown(
                    answer_text
                )

                # ------------------------------------------------
                # Display sources
                # ------------------------------------------------

                if sources_list:

                    st.markdown(
                        "### 📚 المصادر"
                    )

                    for source in sources_list:

                        st.markdown(
                            f"- `{source}`"
                        )

                # ------------------------------------------------
                # Save assistant message
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_list
                    }
                )

            except ValueError as error:

                error_message = (
                    f"⚠️ {str(error)}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )

            except Exception as error:

                error_message = (
                    "⚠️ حدث خطأ أثناء تشغيل النظام. "
                    "يرجى المحاولة مرة أخرى."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )