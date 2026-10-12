"""Streamlit UI. All logic lives in the pdfsum package; this file only wires it up."""

import functools
import io
import time

import streamlit as st

from pdfsum import cache, chunk, config, extract, llm, rag, summarize

st.set_page_config(page_title="Local PDF Summarizer", layout="wide")
st.title("Local PDF Summarizer")
st.caption("Everything runs on your machine through Ollama; your PDF is never uploaded anywhere.")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Chat model", config.SUMMARY_MODEL)
    strategy = st.selectbox(
        "Summary strategy",
        ["map_reduce", "refine", "concat"],
        help="map_reduce: parallel chunk summaries, merged in rounds. "
        "refine: one running summary, updated chunk by chunk. "
        "concat: v1 baseline, chunk summaries pasted together.",
    )
    chunk_size = st.select_slider(
        "Chunk size (characters)", options=[500, 1000, 2000], value=config.CHUNK_SIZE
    )
    use_cache = st.checkbox("Use cached results", value=True)


@st.cache_data(show_spinner="Reading PDF...")
def load_pages(data: bytes):
    return extract.extract_pages(io.BytesIO(data))


@st.cache_data(show_spinner="Running OCR (slow)...")
def load_ocr_pages(data: bytes):
    return extract.ocr_pages(data)


uploaded = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

data = uploaded.getvalue()
file_id = cache.file_hash(data)

try:
    pages = load_pages(data)
except Exception as exc:
    st.error(f"Could not read this PDF: {exc}")
    st.stop()

if extract.looks_scanned(pages):
    st.warning("This PDF has almost no text layer; it is probably a scan.")
    if st.checkbox("Try OCR (needs Tesseract and Poppler installed)"):
        try:
            pages = load_ocr_pages(data)
        except Exception as exc:
            st.error(f"OCR failed: {exc}")
            st.stop()

total_chars = sum(len(p["text"]) for p in pages)
st.success(f"Loaded {len(pages)} pages, {total_chars:,} characters.")
if total_chars == 0:
    st.stop()

summary_tab, ask_tab = st.tabs(["Summary", "Ask this PDF"])

with summary_tab:
    if st.button("Generate summary", type="primary"):
        try:
            llm.check_ollama([model])
        except llm.OllamaUnavailable as exc:
            st.error(str(exc))
            st.stop()

        key = cache.cache_key(file_id, strategy, model, chunk_size)
        result = cache.load(key) if use_cache else None

        if result:
            st.caption("Loaded from cache.")
        else:
            text = "\n".join(p["text"] for p in pages if p["text"])
            chunks = chunk.chunk_text(text, chunk_size, config.CHUNK_OVERLAP)
            llm.STATS.reset()
            bar = st.progress(0.0, text="Summarizing...")

            def on_progress(done: int, total: int) -> None:
                bar.progress(done / total, text=f"Chunk {done} of {total}")

            ask = functools.partial(llm.chat, model=model)
            started = time.time()
            try:
                summary = summarize.summarize(chunks, strategy, llm=ask, progress=on_progress)
            except llm.OllamaUnavailable as exc:
                st.error(str(exc))
                st.stop()
            bar.empty()

            result = {
                "summary": summary,
                "seconds": round(time.time() - started, 1),
                "chunks": len(chunks),
                **llm.STATS.snapshot(),
            }
            if use_cache:
                cache.save(key, result)

        st.subheader("Summary")
        st.markdown(result["summary"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks", result["chunks"])
        c2.metric("Model calls", result["calls"])
        c3.metric("Tokens (prompt + reply)", f"{result['prompt_tokens'] + result['completion_tokens']:,}")
        c4.metric("Seconds", result["seconds"])

        st.download_button(
            "Download summary (.md)", result["summary"], file_name="summary.md", mime="text/markdown"
        )

with ask_tab:
    question = st.text_input("Ask a question about this PDF")
    if question:
        try:
            llm.check_ollama([model, config.EMBED_MODEL])
        except llm.OllamaUnavailable as exc:
            st.error(str(exc))
        else:
            with st.spinner("Indexing the PDF (first time only)..."):
                collection = rag.ingest(file_id, pages, chunk_size)
            hits = rag.retrieve(collection, question)
            st.subheader("Answer")
            st.write_stream(rag.answer_stream(question, hits, model))
            with st.expander("Evidence used (retrieved chunks)"):
                for hit in hits:
                    st.markdown(f"**Page {hit['page']}** (distance {hit['distance']:.3f})")
                    st.write(hit["text"])
