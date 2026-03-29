# app.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from ingest import resolve_channel_id, get_video_count, ingest_channel
from vectorstore import vectorstore_exists, build_vectorstore, load_vectorstore
from retriever import HybridRetriever
from chain import QAChain, SummarizeChain
from config import VIDEO_COUNT_WARNING_THRESHOLD

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YouTube Channel RAG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 YouTube Channel RAG")
st.caption("Turn any YouTube channel into a searchable knowledge base.")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────

if "channel_id" not in st.session_state:
    st.session_state.channel_id = None
if "channel_name" not in st.session_state:
    st.session_state.channel_name = None
if "chains_ready" not in st.session_state:
    st.session_state.chains_ready = False
if "confirmed_large_channel" not in st.session_state:
    st.session_state.confirmed_large_channel = False


# ── Helper: load chains for active channel ────────────────────────────────────

@st.cache_resource
def load_chains(channel_id: str):
    vectorstore = load_vectorstore(channel_id)
    retriever = HybridRetriever(vectorstore=vectorstore)
    return QAChain(retriever=retriever), SummarizeChain(retriever=retriever)


# ── Step 1: Channel input ─────────────────────────────────────────────────────

st.subheader("📺 Step 1: Enter a YouTube Channel")

channel_url = st.text_input(
    "Channel URL or handle",
    placeholder="https://www.youtube.com/@AndrejKarpathy",
)

if st.button("Load Channel", type="primary") and channel_url:

    with st.spinner("Resolving channel..."):
        channel_id = resolve_channel_id(channel_url)

    if not channel_id:
        st.error("Could not resolve channel. Try pasting the full URL.")
        st.stop()

    video_count = get_video_count(channel_id)
    st.session_state.channel_id = channel_id
    st.session_state.video_count = video_count
    st.session_state.chains_ready = False
    st.session_state.confirmed_large_channel = False

    st.success(f"Found channel: `{channel_id}` with **{video_count} videos**")

# ── Step 2: Confirm + ingest ──────────────────────────────────────────────────

if st.session_state.channel_id:
    channel_id = st.session_state.channel_id
    video_count = st.session_state.get("video_count", 0)

    if vectorstore_exists(channel_id):
        # Already ingested — load directly
        st.info(f"✅ Channel already indexed. Loading knowledge base...")
        st.session_state.chains_ready = True

    else:
        # Need to ingest
        if video_count > VIDEO_COUNT_WARNING_THRESHOLD:
            st.warning(
                f"⚠️ This channel has **{video_count} videos** "
                f"(limit: {VIDEO_COUNT_WARNING_THRESHOLD}). "
                f"Ingestion may take several minutes and will cost ~${video_count * 0.003:.2f} "
                f"in OpenAI API calls. Proceed?"
            )
            if st.button("Yes, ingest anyway"):
                st.session_state.confirmed_large_channel = True
        else:
            st.session_state.confirmed_large_channel = True

        if st.session_state.confirmed_large_channel:
            st.subheader("⚙️ Ingesting channel...")

            # Progress bar
            progress_bar = st.progress(0, text="Fetching transcripts...")

            def update_progress(current, total):
                pct = int((current / total) * 100)
                progress_bar.progress(pct, text=f"Processing video {current}/{total}...")

            with st.spinner("Ingesting transcripts..."):
                stats = ingest_channel(
                channel_id=channel_id,
                progress_callback=update_progress,
            )

            progress_bar.progress(100, text="Transcripts fetched!")

            # NEW: guard — don't build if nothing was ingested
            if stats["processed"] == 0:
                st.error(
                    f"No transcripts could be fetched for this channel. "
                    f"All {stats['skipped']} videos were skipped."
                )
                st.stop()

            st.success(
                f"Ingested {stats['processed']} videos "
                f"({stats['skipped']} skipped — no transcript)"
            )

            with st.spinner("Building vector index..."):
                build_vectorstore(channel_id)

            st.success("✅ Knowledge base ready!")
            st.session_state.chains_ready = True

# ── Step 3: Query UI ──────────────────────────────────────────────────────────

if st.session_state.chains_ready:
    channel_id = st.session_state.channel_id
    qa_chain, summarize_chain = load_chains(channel_id)

    st.divider()
    st.subheader("💬 Step 2: Ask Questions")

    with st.sidebar:
        st.header("⚙️ Options")
        mode = st.radio(
            "Query mode",
            options=["Ask a Question", "Summarise a Topic"],
        )
        top_k = st.slider("Chunks to retrieve (k)", 3, 20, 6)
        st.divider()
        if st.button("🔄 Switch Channel"):
            st.session_state.channel_id = None
            st.session_state.chains_ready = False
            st.session_state.confirmed_large_channel = False
            st.cache_resource.clear()
            st.rerun()

    if mode == "Ask a Question":
        query = st.text_input(
            "Your question",
            placeholder="What does this creator say about attention mechanisms?",
        )
        if st.button("Ask", type="primary") and query:
            with st.spinner("Thinking..."):
                result = qa_chain.run(question=query, k=top_k)

            st.markdown("### Answer")
            st.markdown(result["answer"])

            if result["sources"]:
                st.divider()
                st.markdown("### Sources")
                for doc in result["sources"]:
                    title = doc.metadata.get("video_title", "Unknown")
                    t_start = int(doc.metadata.get("chunk_start_time", 0))
                    url = doc.metadata.get("url_with_timestamp", "")
                    minutes, seconds = divmod(t_start, 60)
                    timestamp_str = f"{minutes}:{seconds:02d}"
                    with st.expander(f"📹 {title} — ⏱ {timestamp_str}"):
                        st.markdown(f"[▶ Watch at {timestamp_str}]({url})")
                        st.caption(doc.page_content[:400] + "...")

    else:
        topic = st.text_input(
            "Topic to summarise",
            placeholder="backpropagation",
        )
        if st.button("Summarise", type="primary") and topic:
            with st.spinner(f"Gathering everything about '{topic}'..."):
                result = summarize_chain.run(topic=topic, k=top_k)

            st.markdown("### Summary")
            st.info(f"Synthesised from **{result['video_count']} videos**")
            st.markdown(result["summary"])

            if result["sources"]:
                st.divider()
                st.markdown("### Videos Referenced")
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