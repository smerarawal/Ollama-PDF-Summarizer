# Local LLM PDF Summarizer

A PDF summarization app using:
- Ollama
- Mistral
- Streamlit
- LangChain

## Features
- Upload PDF
- Extract text
- Chunk large documents
- Summarize using local LLM
- Streamlit UI

## Setup

### 1. Install Ollama
Download from:
https://ollama.com

### 2. Pull Model
ollama run mistral

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run Ollama
ollama serve

### 5. Run App
streamlit run app.py
