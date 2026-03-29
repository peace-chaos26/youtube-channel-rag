# app.py
"""
Streamlit UI for YouTube Channel RAG.

Two modes selectable by the user:
  - Ask a Question  : QAChain — specific Q&A with timestamp citations
  - Summarise Topic : SummarizeChain — broad topic synthesis across videos

Run with: streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY from .env

from retriever import HybridRetriever
from chain import QAChain, SummarizeChain
from vectorstore import vectorstore_exists

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Karpathy Knowledge Base",
    page_icon="🧠",
    layout="wide",  
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🧠 Karpathy Knowledge Base")
st.caption(
    "Ask questions across all of Andrej Karpathy's YouTube videos. "
    "Every answer links back to the exact moment in the video."
)
st.divider()

# ── Vectorstore check ─────────────────────────────────────────────────────────

if not vectorstore_exists():
    st.error(
        "⚠️ Vectorstore not found. "
        "Run `python src/vectorstore.py` to build the index first."
    )
    st.stop()

# ── Load retriever + chains (cached — built once per session) ─────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_chains():
    """
    @st.cache_resource ensures the retriever and chains are built once
    per Streamlit session, not on every UI interaction.
    Without this, every button click would reload ChromaDB + rebuild
    the BM25 index — adding ~5s of latency to every query.
    """
    retriever = HybridRetriever()
    return QAChain(retriever=retriever), SummarizeChain(retriever=retriever)

qa_chain, summarize_chain = load_chains()

# ── Sidebar — mode selector + options ─────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Options")

    mode = st.radio(
        "Query mode",
        options=["Ask a Question", "Summarise a Topic"],
        help=(
            "Ask a Question: precise Q&A with citations.\n"
            "Summarise a Topic: broad synthesis across multiple videos."
        ),
    )

    st.divider()

    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=3,
        max_value=20,
        value=6,
        help="More chunks = broader context but higher cost and latency.",
    )

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Built with LangChain · ChromaDB · OpenAI · Streamlit\n\n"
        "Source: [github.com/peace-chaos26/youtube-channel-rag]"
        "(https://github.com/peace-chaos26/youtube-channel-rag)"
    )

# ── Main query area ───────────────────────────────────────────────────────────

if mode == "Ask a Question":

    st.subheader("💬 Ask a Question")
    query = st.text_input(
        "Your question",
        placeholder="What does Karpathy say about attention mechanisms?",
    )

    if st.button("Ask", type="primary") and query:
        with st.spinner("Retrieving and generating answer..."):
            result = qa_chain.run(question=query, k=top_k)

        # ── Answer ────────────────────────────────────────────────────────────
        st.markdown("### Answer")
        st.markdown(result["answer"])

        # ── Sources ───────────────────────────────────────────────────────────
        if result["sources"]:
            st.divider()
            st.markdown("### Sources")
            st.caption(f"{len(result['sources'])} chunks retrieved")

            for doc in result["sources"]:
                title = doc.metadata.get("video_title", "Unknown")
                t_start = int(doc.metadata.get("chunk_start_time", 0))
                url = doc.metadata.get("url_with_timestamp", "")

                # Format timestamp as MM:SS
                minutes, seconds = divmod(t_start, 60)
                timestamp_str = f"{minutes}:{seconds:02d}"

                with st.expander(f"📹 {title} — ⏱ {timestamp_str}"):
                    st.markdown(f"[▶ Watch at {timestamp_str}]({url})")
                    st.caption(doc.page_content[:400] + "...")

else:  # Summarise a Topic

    st.subheader("📚 Summarise a Topic")
    topic = st.text_input(
        "Topic to summarise",
        placeholder="backpropagation",
    )

    if st.button("Summarise", type="primary") and topic:
        with st.spinner(f"Gathering everything Karpathy says about '{topic}'..."):
            result = summarize_chain.run(topic=topic, k=top_k)

        # ── Summary ───────────────────────────────────────────────────────────
        st.markdown("### Summary")
        st.info(f"Synthesised from **{result['video_count']} videos**")
        st.markdown(result["summary"])

        # ── Source videos ─────────────────────────────────────────────────────
        if result["sources"]:
            st.divider()
            st.markdown("### Videos Referenced")

            # Deduplicate by video_id — show one card per video, not per chunk
            seen = set()
            for doc in result["sources"]:
                vid = doc.metadata.get("video_id", "")
                if vid in seen:
                    continue
                seen.add(vid)

                title = doc.metadata.get("video_title", "Unknown")
                date = doc.metadata.get("published_date", "")
                url = f"https://www.youtube.com/watch?v={vid}"

                st.markdown(f"- 📹 [{title}]({url}) — {date}")