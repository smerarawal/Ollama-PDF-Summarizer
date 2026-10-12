# Local PDF Summarizer (v2)

Summarize long PDFs and ask questions about them, entirely on your own machine.
Models run through [Ollama](https://ollama.com); the PDF never leaves your computer.

## What it does

- **Summaries of long documents** with three strategies you can compare:
  `map_reduce` (parallel chunk summaries merged in rounds), `refine` (one running
  summary updated chunk by chunk) and `concat` (the v1 behaviour, kept as a baseline).
- **Ask this PDF**: retrieval-augmented answers that cite page numbers, with the
  retrieved chunks shown as evidence.
- Page-aware extraction, a warning (and optional OCR) for scanned PDFs, a cache so a
  document is never summarized twice with the same settings, progress bar, token and
  time counters, and clear errors when Ollama is off or a model is missing.

## Architecture

```
PDF --pypdf--> pages [{page, text}]
   |
   |-- summary: join text -> chunk (1000 chars, 200 overlap) -> strategy -> structured summary
   |
   '-- ask:     chunk per page -> embed (nomic-embed-text) -> ChromaDB (cosine)
                question -> embed -> top 4 chunks -> prompt "[p.N] text" -> cited answer
```

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI only |
| `pdfsum/extract.py` | Text per page, scanned-PDF detection, optional OCR |
| `pdfsum/chunk.py` | Chunking for summaries (whole text) and for retrieval (per page) |
| `pdfsum/llm.py` | Ollama wrapper: fixed options, retries, token counts, friendly errors |
| `pdfsum/summarize.py` | `concat`, `map_reduce`, `refine` (each takes an `llm` callable, so it is testable) |
| `pdfsum/rag.py` | ChromaDB ingestion, retrieval, cited question answering |
| `pdfsum/cache.py` | JSON cache keyed by file hash, strategy, model and chunk size |
| `pdfsum/config.py` | All settings, overridable with environment variables |
| `legacy/ollamapdf.py` | The original v1 script, kept for reference |

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:

   ```
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

2. Create a virtual environment and install the dependencies:

   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # macOS / Linux
   pip install -r requirements-dev.txt
   ```

3. Make sure Ollama is running (the desktop app starts it automatically; otherwise run
   `ollama serve` in another terminal), then start the app:

   ```
   streamlit run app.py
   ```

## Configuration

Set environment variables before starting, for example `set SUMMARY_MODEL=llama3.1` on Windows.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SUMMARY_MODEL` | `mistral` | Chat model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `NUM_CTX` | `8192` | Context window sent with every call |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Characters per chunk and overlap |
| `REDUCE_LIMIT` | `6000` | Merge summaries until they total fewer characters than this |
| `WORKERS` | `3` | Parallel model calls (start Ollama with `OLLAMA_NUM_PARALLEL=3` to benefit) |
| `TOP_K` | `4` | Chunks retrieved per question |

## Tests

```
python -m pytest -q
```

The tests use a stand-in model and stand-in embeddings, so they run without Ollama.
They cover the chunker, the reduce-grouping logic (including a proof that it always
terminates), all three strategies, the cache, retrieval returning the right page, and
the Ollama wrapper's retries and error messages.

## Design decisions

- **Chunk for summaries by whole text, chunk for retrieval by page.** Summaries want few,
  full chunks (fewer model calls); citations need an exact page number per chunk.
- **Hierarchical reduce.** One final prompt over hundreds of chunk summaries would overflow
  the context window, so summaries are merged in rounds until they fit.
- **`pack()` guarantees progress.** Each merge group has at least two items, so every round
  shrinks the list and the loop cannot run forever.
- **`num_ctx` and temperature are sent on every call**, so a server restart cannot silently
  change them.
- **Cosine distance** for retrieval; Chroma's default is squared L2.
- **Dependency injection** of the model callable keeps strategy logic unit-testable.

## Limitations

- Small local models can still hallucinate; the prompts restrict them to the supplied text
  but do not guarantee it.
- Scanned PDFs need the optional OCR path (Tesseract and Poppler installed).
- On CPU, a long document takes minutes; the cache makes repeat runs instant.
- Tables and multi-column layouts may extract in a confusing order (a pypdf limitation).

## Results

Evaluation of the three strategies, chunk sizes and models is added after Day 2
(`eval/`), with ROUGE, BERTScore, latency and a small human rating.
