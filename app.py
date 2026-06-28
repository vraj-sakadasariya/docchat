import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

# works locally with .env, and on Streamlit Cloud with their secrets manager
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


st.set_page_config(
    page_title="DocChat",
    page_icon="document",
    layout="wide",
)

# basic styling - keeping it simple
st.markdown("""
<style>
    #MainMenu, footer { visibility: hidden; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .source-box {
        background-color: #1e1e2e;
        border-left: 3px solid #7c3aed;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #cdd6f4;
        line-height: 1.5;
    }

    .score-label {
        font-size: 0.75rem;
        color: #7c3aed;
        margin-bottom: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# session state setup
if "engine" not in st.session_state:
    st.session_state.engine = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# sidebar
with st.sidebar:
    st.title("DocChat")
    st.caption("Ask questions about any document")
    st.divider()

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

    if uploaded_file:
        file_bytes = uploaded_file.read()

        # only reprocess if it's a new file
        if (st.session_state.doc_stats is None or
                st.session_state.doc_stats.get("doc_name") != uploaded_file.name):

            with st.spinner("Processing document..."):
                try:
                    engine = RAGEngine(api_key=GROQ_API_KEY)
                    stats = engine.load_document(file_bytes, uploaded_file.name)

                    st.session_state.engine = engine
                    st.session_state.doc_stats = stats
                    st.session_state.messages = []
                    st.session_state.last_sources = []

                    st.success("Done! Document is ready.")
                except Exception as e:
                    st.error(f"Failed to process document: {e}")
                    st.stop()

    if st.session_state.doc_stats:
        st.divider()
        stats = st.session_state.doc_stats
        st.markdown(f"**File:** {stats['doc_name']}")

        col1, col2 = st.columns(2)
        col1.metric("Words", f"{stats['words']:,}")
        col2.metric("Chunks", stats["chunks"])

        if st.button("Clear and start over", use_container_width=True):
            st.session_state.engine = None
            st.session_state.doc_stats = None
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

    st.divider()
    with st.expander("How it works"):
        st.markdown("""
1. Your document gets split into small chunks
2. Each chunk is converted into a vector (embedding)
3. When you ask a question, the most relevant chunks are retrieved
4. The LLM answers using only those chunks

This approach is called RAG — Retrieval-Augmented Generation.
        """)


# main area
if st.session_state.engine is None:
    st.header("DocChat")
    st.write("Upload a PDF, Word doc, or text file from the sidebar to get started.")
    st.info("The AI will only answer based on what's in your document — no hallucinations.")
else:
    # show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # show sources from last response
    if st.session_state.last_sources:
        with st.expander("View sources used for last answer"):
            for i, (chunk, score) in enumerate(st.session_state.last_sources, 1):
                st.markdown(f"""
<div class="source-box">
    <div class="score-label">Excerpt {i} — relevance: {int(score * 100)}%</div>
    {chunk[:400]}{"..." if len(chunk) > 400 else ""}
</div>
""", unsafe_allow_html=True)

    # chat input
    user_input = st.chat_input(f"Ask something about {st.session_state.doc_stats['doc_name']}...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                stream, sources = st.session_state.engine.ask(
                    query=user_input,
                    chat_history=history,
                )

                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    full_response += token
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.last_sources = sources

            except Exception as e:
                placeholder.error(f"Error: {e}")

        st.rerun()
