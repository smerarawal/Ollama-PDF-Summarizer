"""Central settings. Every value can be overridden with an environment variable."""

import os

# Models (pull them first: `ollama pull mistral` and `ollama pull nomic-embed-text`)
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "mistral")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Generation settings. num_ctx is passed with every call so a restart of the
# Ollama server cannot silently shrink the context window.
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Chunking (characters). 1000 characters is roughly 250 tokens.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Map-reduce: keep merging summaries until their combined length is below this.
REDUCE_LIMIT = int(os.getenv("REDUCE_LIMIT", "6000"))
WORKERS = int(os.getenv("WORKERS", "3"))

# Retrieval for "Ask this PDF"
TOP_K = int(os.getenv("TOP_K", "4"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))

# Storage
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
