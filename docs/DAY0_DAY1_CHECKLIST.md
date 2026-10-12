# Day 0 and Day 1 checklist

Tick items off in order. Estimated time: Day 0 about 2 hours, Day 1 about 5 to 6 hours.

---

## Day 0A: PDF Summarizer repo (`Ollama-PDF-Summarizer`) — 30 min

The v2 project in this folder replaces the old repo contents, so tag the old code first.

```
git clone https://github.com/smerarawal/Ollama-PDF-Summarizer
cd Ollama-PDF-Summarizer

git tag v1-baseline
git push origin v1-baseline

git checkout -b v2
mkdir legacy
git mv ollamapdf.py legacy/ollamapdf.py
git rm requirements.txt README.md .gitignore
```

Now copy **everything inside the `pdf-summarizer` folder** (app.py, pdfsum/, tests/, docs/,
eval/, README.md, requirements.txt, requirements-dev.txt, pytest.ini, .gitignore) into the
repo root, then:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
ollama pull mistral
ollama pull nomic-embed-text
python -m pytest -q
```

- [ ] `python -m pytest -q` ends with `37 passed`
- [ ] `git add -A && git commit -m "v2: restructure into pdfsum package, add map-reduce, refine, cited Q&A, tests"`

Why this order: `v1-baseline` is the code your v2 numbers are compared against on Day 2.

---

## Day 0B: Research Analyst repo (`Agentic-Research-Analyst-by-Ollama`) — 1.5 hours

Do these fixes on `main` **first**, then tag, so the tag is a clean, honest v1.

```
git clone https://github.com/smerarawal/Agentic-Research-Analyst-by-Ollama
cd Agentic-Research-Analyst-by-Ollama
```

**1. Remove the dead LangGraph file** (it uses `planner_agent` and `critic_agent`, which do not exist):

```
git rm workflow.py
```

**2. Align models with the README.** In `crew/agents.py` change both the researcher and the
critic from `llm="ollama/llama3"` to `llm="ollama/llama3.1"`, then:

```
ollama pull llama3.1
```

**3. Fix the keyword bug in `agents/search_agent.py`.** The test `word in combined` matches
"ai" inside "said" or "email". Add `import re` at the top and replace the keyword check with:

```python
            if any(
                re.search(rf"\b{re.escape(word)}\b", combined)
                for word in important_keywords
            ):
                score += 2
```

**4. Make the UI text honest.** In `app.py` replace the line
`- A2A communication via task context` with
`- Task context passed between agents (CrewAI)`.

**5. Fix the README.**
- Replace `git clone https://github.com/your-username/agentic-research-analyst` and the
  `cd` line with `https://github.com/smerarawal/Agentic-Research-Analyst-by-Ollama` and
  `cd Agentic-Research-Analyst-by-Ollama`.
- Delete the **second** "Built with" block (the one that lists LangGraph); it comes back on Day 3.
- Add `agents/search_agent.py   # DuckDuckGo search and result ranking` to the project-structure tree.

**6. Stop committing generated files.**

```
mkdir examples
git mv research_report.txt examples/sample_report.txt
```

Then open `.gitignore` in an editor and add a line `research_memory.json` (do not append
with `echo >>` in PowerShell; it can save the file in an encoding Git cannot read).

**7. Check dependencies and unused files.**
- Open `pyproject.toml`: it must list every package you import (crewai, streamlit, chromadb,
  ddgs, beautifulsoup4, requests, ollama if used).
- Look at `main.py`, `utils/ollama_client.py`, `utils/json_parser.py`, `utils/drive_saver.py`:
  if nothing imports a file, delete it or say in the README what it is for.

**8. Prove v1 still runs from a clean clone**, then freeze it:

```
uv sync
set OLLAMA_NUM_CTX=16384
ollama serve        (skip if the Ollama app is already running)
streamlit run app.py
```

Run one specific query, e.g. `NVIDIA Blackwell GPU data center deployment 2024`, and save the
resulting report. Then:

```
git add -A
git commit -m "chore: align models, fix keyword matching, remove dead code, fix README"
git tag v1-baseline
git push origin main v1-baseline
git checkout -b v2-langgraph
```

- [ ] A fresh clone runs from the README alone
- [ ] `v1-baseline` tag pushed
- [ ] You are on branch `v2-langgraph`

---

## Day 1: run and understand PDF Summarizer v2

### 1. First run (30 min)

```
ollama serve        (skip if the Ollama app is already running)
streamlit run app.py
```

- [ ] Upload a normal text PDF (10 to 20 pages). Click **Generate summary** with `map_reduce`.
      You should see a progress bar, a structured summary (Overview, Key Points, Conclusion)
      and metrics for chunks, model calls, tokens and seconds.
- [ ] Click **Generate summary** again: the caption says "Loaded from cache" and it is instant.
- [ ] Open **Ask this PDF**, ask a question whose answer is on a known page. The answer
      cites `[p.N]`; open "Evidence used" and check page N really says that.
- [ ] Ask something that is **not** in the PDF. The model should say it cannot find it.

### 2. Break it on purpose (30 min)

Each of these must give a clear message, not a traceback.

- [ ] Stop Ollama, click Generate: message says to run `ollama serve`
- [ ] Type a model you have not pulled (e.g. `llama3.1` before pulling it): message says to run `ollama pull llama3.1`
- [ ] Upload a scanned PDF (or a PDF of photos): warning about no text layer
- [ ] Upload a password-protected PDF: "Could not read this PDF: This PDF is password protected."

### 3. First measurements (45 min) — these feed Day 2

On **one** 20-page PDF, write down seconds, calls and tokens for each strategy:

| Strategy | Chunks | Model calls | Tokens | Seconds | Summary length (words) |
| --- | --- | --- | --- | --- | --- |
| concat | | | | | |
| map_reduce | | | | | |
| refine | | | | | |

Untick "Use cached results" between runs. If you can, restart Ollama with
`set OLLAMA_NUM_PARALLEL=3` and compare map_reduce with and without it. Note what you
observe, including if parallel calls give no speed-up on your hardware.

### 4. Read the code until you can explain it (2 hours)

Read in this order, and for each file say its job in one sentence without looking:

1. `pdfsum/config.py`: where every setting lives
2. `pdfsum/extract.py`: why empty pages are kept, not dropped (page numbers must match the PDF)
3. `pdfsum/chunk.py`: why summaries chunk the whole text but retrieval chunks per page
4. `pdfsum/llm.py`: what the retries, `STATS` and `check_ollama` are for
5. `pdfsum/summarize.py`: trace `map_reduce` on paper with 8 chunks and `limit` small; read `pack()` and explain why the loop must terminate
6. `pdfsum/rag.py`: ingest, retrieve, build_prompt; why cosine distance
7. `app.py`: how the pieces connect

Self-test (answer aloud):

- [ ] Why does v1 produce a pile of mini-summaries on a long PDF, and how does v2 fix it?
- [ ] What happens if the merged summaries are still too long for one prompt?
- [ ] Why is `llm` passed in as an argument instead of imported inside the strategies?
- [ ] What is the difference between `chunk_text` and `chunk_pages`, and why have both?
- [ ] Why is the collection named by file hash and chunk size?
- [ ] What does `num_ctx` control and why is it sent on every call?

### 5. Commit and push (15 min)

```
git add -A
git commit -m "docs: add Day 0/1 notes and first strategy measurements"
git push -u origin v2
```

Add a screenshot of the Summary tab and of a cited answer to the README (put the images in `docs/`).

### Done when

- [ ] `python -m pytest -q` passes
- [ ] You ran all three strategies on a real PDF and filled in the table above
- [ ] A cited answer's page number matches the PDF
- [ ] Every "break it on purpose" case shows a clear message
- [ ] You can answer the six self-test questions without looking

If something fails, send me the exact error text or screenshot and the PDF size, and I will fix it.
