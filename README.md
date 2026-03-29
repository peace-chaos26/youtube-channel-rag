# 🧠 YouTube Channel RAG

> Turn any YouTube channel into a searchable knowledge base.
> Ask questions across hundreds of videos — every answer links back
> to the exact moment in the video.

![Demo](assets/demo.gif)

---

## What It Does

Point this at any YouTube channel and ask natural language questions
across the entire video library — without watching a single video.

**Example queries:**
- *"What does Karpathy say about attention mechanisms?"*
- *"Summarise everything across all videos about backpropagation"*
- *"How does nanoGPT differ from the original GPT-2?"*

Every answer cites the exact video and timestamp:
`Let's build GPT — ▶ Watch at 7:12`

---

## Architecture
```
YouTube Channel
      ↓
  ingest.py        Fetch transcripts + metadata for every video
      ↓
  chunker.py       Merge 3s segments → 512-token chunks, preserve timestamps
      ↓
  vectorstore.py   Embed (text-embedding-3-small) → ChromaDB
      ↓
  retriever.py     Hybrid retrieval: BM25 + dense → Reciprocal Rank Fusion
      ↓
  chain.py         QA chain (single LLM call) or Map-Reduce summarisation
      ↓
  app.py           Streamlit UI with timestamp-linked citations
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Chunking | Fixed-size 512t, 64t overlap | Transcript text is conversational; fixed-size with overlap handles boundary splits |
| Retrieval | Hybrid BM25 + dense | Dense fails on exact terms (nanoGPT); BM25 fails on semantic queries; hybrid covers both |
| Fusion | Reciprocal Rank Fusion | Scores from BM25 and cosine similarity are on incompatible scales; RRF uses ranks instead |
| Summarisation | Map-Reduce | Topic may span 20+ videos; single LLM call would exceed context limits |
| Vector DB | ChromaDB | Zero-infra, persistent, sufficient for ~60 video corpus |
| Embeddings | text-embedding-3-small | 5× cheaper than large; quality difference marginal at this corpus size |

---

## Evaluation

Evaluated on a hand-crafted set of 15 Q&A pairs with known ground truth.

| Metric | Score |
|---|---|
| Retrieval Precision@6 | 0.53 |
| Mean Faithfulness (LLM-as-judge) | 4.6 / 5 |

Faithfulness judged by `gpt-4o` evaluating `gpt-4o-mini` outputs
to avoid self-evaluation bias.

Run evals yourself:
```bash
python src/eval.py
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Transcript ingestion | `youtube-transcript-api`, `scrapetube`, `yt-dlp` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB |
| Sparse retrieval | `rank-bm25` |
| LLM | OpenAI `gpt-4o-mini` |
| Orchestration | LangChain |
| UI | Streamlit |

---

## Getting Started

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation
```bash
git clone https://github.com/peace-chaos26/youtube-channel-rag.git
cd youtube-channel-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

### Run the pipeline
```bash
# Step 1 — fetch all transcripts (run once)
python src/ingest.py

# Step 2 — build the vector index (run once)
python src/vectorstore.py

# Step 3 — launch the app
streamlit run app.py
```

### Run evals
```bash
python src/eval.py
```

---

## Project Structure
```
youtube-channel-rag/
├── src/
│   ├── config.py         Constants and metadata field definitions
│   ├── ingest.py         YouTube → transcript JSON files
│   ├── chunker.py        Transcript chunks with timestamp preservation
│   ├── vectorstore.py    ChromaDB build and load
│   ├── retriever.py      Hybrid BM25 + dense retrieval with RRF
│   ├── chain.py          QA chain + Map-Reduce summarisation
│   └── eval.py           Precision@k + faithfulness evaluation
├── app.py                Streamlit UI
├── eval_data/            Hand-crafted eval set + results
├── notebooks/            Exploration
├── .env.example
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Semantic chunking (topic-boundary splits)
- [ ] Multi-hop retrieval for complex questions
- [ ] Deploy on Hugging Face Spaces
- [ ] Extend to multi-channel corpus
- [ ] RAGAS integration for richer eval metrics

---

## License

MIT